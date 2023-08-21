from collections.abc import Sequence
import torch

from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, multiclass_nms
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models import DETECTORS, build_detector

from acdet.utils.structure_utils import dict_split, weighted_loss
from acdet.utils import log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid


@DETECTORS.register_module()
class MixedTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        teacher = build_detector(model)
        student = build_detector(model)
        super(MixedTeacher, self).__init__(
            dict(teacher=teacher, student=student),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
            self.partial_weight = self.train_cfg.get('partial_weight')
            self.partial_boxmerge_iou_thresh = self.train_cfg.partial_boxmerge_iou_thresh
            self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        elif test_cfg is not None:  # only applies during test
            self.unlabel_weight = test_cfg.unlabel_weight
            self.train_cfg = test_cfg.aug_cfg

    def forward_test(self, imgs, img_metas, **kwargs):
        if kwargs.get("active_cycle") is not None:
            self.active_cycle = kwargs.pop("active_cycle")
            return self.active_test(imgs, img_metas, **kwargs)
        else:
            return self.model(**kwargs).forward_test(imgs, img_metas, **kwargs)

    def active_test(self, img, img_metas, **kwargs):
        # print("Actively Evaluating ...")
        rescale = kwargs.pop("rescale")
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")
        teacher_data, student_data = data_groups["unlabel_weak"], data_groups["unlabel_strong"]
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)
        if True and teacher_data.get("gt_bboxes") is not None:
            teacher_info["gt_bboxes"] = teacher_data["gt_bboxes"]
            teacher_info["gt_labels"] = teacher_data["gt_labels"]

        return self.compute_active_scores(student_info, teacher_info)

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "labeled" in data_groups:
            gt_bboxes = data_groups["labeled"]["gt_bboxes"]
            log_every_n(
                {"labeled_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["labeled"])
            sup_loss = {"L_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unlabel_strong" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unlabel_weak"], data_groups["unlabel_strong"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"U_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
        if "partial_strong" in data_groups:
            punsup_loss = weighted_loss(
                self.foward_partial_train(
                    data_groups["partial_weak"], data_groups["partial_strong"]
                ),
                weight=self.partial_weight,
            )
            punsup_loss = {"P_" + k: v for k, v in punsup_loss.items()}
            loss.update(**punsup_loss)
        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)
        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def foward_partial_train(self, teacher_data, student_data):
        # merge labeled GTs and pseudo labels and then train as unlabeled
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)
        # Merge
        merged_bboxes, merged_labels = self.merge_gt_pseudo(
            teacher_data["gt_bboxes"], 
            teacher_data["gt_labels"], 
            teacher_info["det_bboxes"],
            teacher_info["det_labels"]
        )
        teacher_info["det_bboxes"] = merged_bboxes
        teacher_info["det_labels"] = merged_labels
        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def merge_gt_pseudo(self, gt_bboxes, gt_labels, pseudo_bboxes, pseudo_labels):
        merged_bboxes = []
        merged_labels = []
        for img_i, (gt_bbox, gt_label, pseudo_bbox, pseudo_label) in \
                enumerate(zip(gt_bboxes, gt_labels, pseudo_bboxes, pseudo_labels)):
            pseudo_score = None
            if pseudo_bbox.shape[1] > 4:
                pseudo_bbox, pseudo_score = pseudo_bbox.split([4, pseudo_bbox.shape[1] - 4], dim=1)
            overlaps = self.iou_calculator(gt_bbox, pseudo_bbox)
            iou_filter = overlaps > self.partial_boxmerge_iou_thresh    # (m, n)
            final_filter = iou_filter

            unlabel_idxs = torch.sum(final_filter, 0) == 0
            merged_box = torch.cat([gt_bbox, pseudo_bbox[unlabel_idxs]])
            merged_label = torch.cat([gt_label, pseudo_label[unlabel_idxs]])
            if pseudo_score is not None:
                gt_score = pseudo_score.new_zeros((gt_bbox.shape[0], pseudo_score.shape[1]))
                gt_score[:, 0] = gt_score.new_ones(gt_score.shape[0])
                merged_score =  torch.cat([gt_score, pseudo_score[unlabel_idxs]], dim=0)
            merged_bboxes.append(torch.cat([merged_box, merged_score], dim=1))
            merged_labels.append(merged_label)
        return merged_bboxes, merged_labels

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        # pseudo_boxes (:, 5) composed of 4positions, 1reg_unc
        teacher_det_boxes = teacher_info["det_bboxes"]
        pseudo_bboxes = self._transform_bbox(
            teacher_det_boxes,
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        
        pseudo_bboxes_rpn, _, _, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.rpn_pseudo_threshold,
            min_size=self.train_cfg.min_pseduo_box_size,
        )
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes_rpn,
            student_info["img_metas"],
        )
        loss.update(rpn_loss)

        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        pseudo_bboxes_cls, pseudo_labels_cls, _, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        loss_cls, _ = self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes_cls,
                pseudo_labels_cls,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        loss.update(loss_cls)
        pseudo_bboxes_reg, pseudo_labels_reg, _, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        loss_reg, _ = self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes_reg,
                pseudo_labels_reg,
                student_info=student_info,
            )
        loss.update(loss_reg)
        return loss

    def compute_active_scores(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        teacher_det_boxes = teacher_info["det_bboxes"]
        pseudo_bboxes = self._transform_bbox(
            teacher_det_boxes,
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        
        ori_shape = teacher_info["img_metas"][0]["ori_shape"]
        inv_teacher_det_boxes = [
            Transform2D.transform_bboxes(v, 
                teacher_info['transform_matrix'][i].inverse(), 
                ori_shape) 
            for i, v in enumerate(teacher_det_boxes)]
        merged_bboxes = torch.stack(inv_teacher_det_boxes).mean(dim=0)[:, :5]
        merged_labels = torch.mode(torch.stack(pseudo_labels), 0)[0]
        aug_times = len(inv_teacher_det_boxes)
        all_scores = merged_bboxes.new_zeros((aug_times, merged_bboxes.shape[0], 2))
        all_scores[all_scores==0] = float('nan')
        
        pseudo_bboxes_rpn, _, _, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.rpn_pseudo_threshold,
            min_size=self.train_cfg.min_pseduo_box_size,
            reverse=self.train_cfg.get('rev_thr', False)
        )
        _, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes_rpn,
            student_info["img_metas"],
            student_info=student_info,
        )

        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        # classification disagreement
        pseudo_bboxes_cls, pseudo_labels_cls, _, pseudo_idx_cls = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
            reverse=self.train_cfg.get('rev_thr', False)
        )
        _, pos_assigned_gt_scores_cls = self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes_cls,
                pseudo_labels_cls,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        for aug_i, (x, xscore) in enumerate(zip(pseudo_idx_cls, pos_assigned_gt_scores_cls)):
            all_scores[aug_i, x, 0] = xscore.float()
        # torch.nanmean()
        all_scores = torch.div(all_scores.nansum(dim=0), (~torch.isnan(all_scores)).count_nonzero(dim=0))

        pseudo_bboxes_reg, pseudo_labels_reg, _, pseudo_idx_reg = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
            reverse=self.train_cfg.get('rev_thr', False)
        )
        if pseudo_bboxes_reg[0].shape[0]:
            un_reg = self.unsup_rcnn_reg_scores(
                    student_info["img_metas"],
                    proposals,
                    pseudo_bboxes_reg,
                    pseudo_labels_reg,
                    teacher_info["transform_matrix"],
                    student_info["transform_matrix"],
                    teacher_info["img_metas"],
                    teacher_info["backbone_feature"],
                )
            un_reg_t = torch.cat(un_reg, dim=0)
            if un_reg_t.shape[0] != pseudo_idx_reg[0].shape[0]:
                mask = un_reg_t.new_ones(pseudo_idx_reg[0].numel(), dtype=torch.bool)
                for i, un_reg_i in enumerate(un_reg):
                    if not un_reg_i.shape[0]:
                        mask[i] = False
                pseudo_idx_reg[0] = pseudo_idx_reg[0][mask]
            all_scores[pseudo_idx_reg[0], 1] = un_reg_t.mean(dim=1)
        
        all_scores = all_scores.detach().cpu().numpy()

        res = []
        for box_idx, box in enumerate(merged_bboxes):
            res_box = dict(
                bbox=box[:4],
                score=float(box[4].detach().cpu()),
                category_id=int(merged_labels[box_idx].detach().cpu()),
                score_cls=all_scores[box_idx, 0],
                score_reg=all_scores[box_idx, 1],
                )
            res.append(res_box)
        return [res]

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in pseudo_bboxes]) / len(pseudo_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in pseudo_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in pseudo_bboxes]) / len(pseudo_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
        ) 
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, pseudo_bboxes, pseudo_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        num_classes = self.student.roi_head.bbox_head.num_classes
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
            **kwargs
        )  
        if loss.get("loss_cls") is None:
            loss["loss_cls"] = rois.new_zeros(())
        if hasattr(self, "active_cycle"):
            loss["loss_cls"] = loss["loss_cls"] / max(bbox_targets[1].sum(), 1.0)
            pos_assigned_gt_scores = []
            vis_bboxes = []
            vis_labels = []
            splits = [res.bboxes.shape[0] for res in sampling_results]
            losses = loss["loss_cls"].split(splits)
            labels = bbox_targets[0].split(splits)
            rois = rois.split(splits, 0)
            boxes = [self.student.roi_head.bbox_head.bbox_coder.decode( \
                rois[i][..., 1:], b, max_shape=student_info["img_metas"][i]['img_shape']).view(b.size(0), -1, 4) \
                    for i, b in enumerate(bbox_results["bbox_pred"].split(splits))]
            scores = bbox_results["cls_score"].split(splits)
            for i, res in enumerate(sampling_results):
                pos_assigned_gt_score = []
                vis_label = []
                vis_bbox = []
                for gt_ind in range(res.num_gts):
                    gt_indices = (res.pos_assigned_gt_inds == gt_ind).nonzero().squeeze()
                    pos_assigned_gt_score.append(losses[i][gt_indices].sum().detach())
                    pos_assigned_gt_labels = labels[i][gt_indices].detach().view(-1)
                    vis_label.append(scores[i][gt_indices].view(-1, num_classes+1)[:, :num_classes].argmax(dim=1).detach().view(-1))
                    vis_bbox.append(torch.cat([boxes[i][gt_indices, pos_assigned_gt_labels].detach().view(-1, 4), losses[i][gt_indices].view(-1, 1).detach()], dim=-1))
                pos_assigned_gt_scores.append(torch.stack(pos_assigned_gt_score, 0) if pos_assigned_gt_score \
                    else torch.Tensor(pos_assigned_gt_score).to(losses[0].device).float())
                if len(vis_label):
                    vis_labels.append(torch.cat(vis_label, dim=0))
                    vis_bboxes.append(torch.cat(vis_bbox, dim=0))
            loss["loss_cls"] = loss["loss_cls"].sum()
        else:
            loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
            pos_assigned_gt_scores = None
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        return loss, pos_assigned_gt_scores

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in pseudo_bboxes]) / len(pseudo_bboxes)}
        )
        if hasattr(self, "active_cycle") and True:
            kwargs['reduction_override'] = 'none'
            loss_bbox, pos_assigned_gt_scores = self.student.roi_head.forward_train(
                feat, img_metas, proposal_list, pseudo_bboxes, pseudo_labels, **kwargs
            )
        else:
            loss_bbox = self.student.roi_head.forward_train(
                feat, img_metas, proposal_list, pseudo_bboxes, pseudo_labels, **kwargs
            )
            pos_assigned_gt_scores = None
        # loss_bbox (pos_ind, 4) if none reduction
        return {"loss_bbox": loss_bbox["loss_bbox"]}, pos_assigned_gt_scores

    def unsup_rcnn_reg_scores(
        self,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        **kwargs,
    ):
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in pseudo_bboxes]) / len(pseudo_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
        )   # sample 512 cat(pos, neg) boxes from 1K proposals 
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        num_gts = sampling_results[0].num_gts
        matched_proposal_list = [[] for _ in range(num_gts)]
        if num_gts:
            for i, res in enumerate(sampling_results):
                if res.num_gts == num_gts:
                    for gt_ind in range(res.num_gts):
                        gt_indices = (res.pos_assigned_gt_inds == gt_ind).nonzero().squeeze(1)
                        matched_proposal_list[gt_ind].append(aligned_proposals[i][gt_indices].detach())
            matched_proposal_list = [torch.cat(mp, 0) for mp in matched_proposal_list]       
        jitter_times = [pl.shape[0] for pl in matched_proposal_list]
        teacher_feat = [torch.stack([tf[0]]*len(matched_proposal_list), 0) for tf in teacher_feat]
        teacher_img_metas = [teacher_img_metas[0]] * len(matched_proposal_list)
        gt_label_list = pseudo_labels[0].split(1)
        reg_unc = self.compute_uncertainty_with_aug(
            teacher_feat, teacher_img_metas, matched_proposal_list, gt_label_list, jitter_times
        ) 
        return reg_unc

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        teacher_info["img"] = img
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _, proposal_valid_idx = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, auged_proposal_list, proposal_label_list, self.train_cfg.jitter_times
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug(
        self, feat, img_metas, auged_proposal_list, proposal_label_list, jitter_times
    ):
        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4     # cls-specific reg
        if not isinstance(jitter_times, Sequence):
            jitter_times = [jitter_times] * len(bboxes)
        bboxes = [
            bbox.reshape(jt, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(jt, 0, 4 * reg_channel).float()
            for bbox, jt in zip(bboxes, jitter_times)
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
