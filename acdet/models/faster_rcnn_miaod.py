import torch
import torch.nn as nn

from mmcv.runner import force_fp32

from mmdet.core import bbox2roi, multiclass_nms
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import FasterRCNN

from acdet.utils.structure_utils import dict_split

@DETECTORS.register_module()
class FasterRCNNMIAOD(FasterRCNN):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNMIAOD, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.loss_type = -1

    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")
        if self.loss_type == 0:
            # labeled cls + loc + imgcls
            losses = super().forward_train(
                **data_groups["labeled"])
            loss_names_exclude = ["unlabel_loss_wavedis", "unlabel_loss_imgcls"]
        else:
            losses = super().forward_train(
                **data_groups["labeled"])
            cls_score, bbox_pred = self.forward_train_unlabeled(
                **data_groups["unlabeled"])
            if self.loss_type == 1:
                # labeled cls + loc + imgcls, unlabeled l_wave_dis + imgcls
                loss_names_exclude = []
                losses_unlabeled = self.roi_head.bbox_head.loss_wave_min(*cls_score)
                losses.update(losses_unlabeled)
            elif self.loss_type == 2:
                # labeled cls + loc, unlabeled l_wave_dis_minus
                loss_names_exclude = ["loss_imgcls", "unlabel_loss_imgcls"]
                losses_unlabeled = self.roi_head.bbox_head.loss_wave_dis(*cls_score, True)
                losses.update(losses_unlabeled)
            else:
                raise NotImplementedError
        for en in loss_names_exclude:
            losses[en] = losses['loss_cls'].new_zeros(1)[0]
        return losses

    def forward_train_unlabeled(self,
                                img,
                                img_metas,
                                gt_bboxes,
                                gt_labels,
                                gt_bboxes_ignore=None,
                                gt_masks=None,
                                proposals=None,
                                **kwargs):
        x = self.extract_feat(img)
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            _, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.roi_head.with_bbox or self.roi_head.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_feats = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        if self.roi_head.with_shared_head:
            bbox_feats = self.roi_head.shared_head(bbox_feats)
        cls_score, bbox_pred = self.roi_head.bbox_head(bbox_feats)
        return cls_score, bbox_pred

    def forward_test(self, imgs, img_metas, **kwargs):
        if kwargs.get("active_cycle") is not None:
            self.active_cycle = kwargs.pop("active_cycle")
            return self.simple_test_scores(imgs, img_metas, **kwargs)
        else:
            return super().forward_test(imgs, img_metas, **kwargs)

    def simple_test_scores(self, imgs, img_metas, proposals=None, rescale=False):
        """Test without augmentation and return predicted scores instead of bboxes."""

        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if proposals is not None:
                proposals = proposals[0]
            img_metas = img_metas[0]
            
            assert self.with_bbox, 'Bbox head must be implemented.'
            x = self.extract_feat(imgs[0])
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            return self.roi_head.simple_test_scores(
                x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)
        else:
            raise NotImplementedError

