import json
import math
import torch
import random
import os.path as osp

from mmdet.core.bbox.assigners import MaxIoUAssigner

from ..utils import *
from ..registry import MINERS
from .mining_algorithm import MiningAlgorithm

@MINERS.register_module(name='box_compas')
class ComPASMiningAlgorithm(MiningAlgorithm):
    def __init__(self, logger, config):
        super(ComPASMiningAlgorithm, self).__init__(
            logger, config
        )
        self.weight = dict(score_cls=1, score_run=0, score_reg=1)

    def run(self, labeled_data, partial_data, ratio):
        score_thresh = self.config.get("score_thresh", 0)
        pred_data = self.load(score_thresh)
        labeled_data, remained_data = self.mining(labeled_data, partial_data, pred_data, ratio)
        return labeled_data, remained_data

    def load(self, score_thresh):
        if len(self.config.model_result) == 1:
            self.config.model_result = self.config.model_result[0]
        box_list = json.load(open(self.config.model_result, 'r'))
        return box_list

    def value(self, data, scale_factors=None, weight=None):
        if not data['bbox']:
            return np.inf
        vs = []
        ws = []
        for k, v in data.items():
            if k.startswith('score_'):
                w = self.weight[k]
                if w > 0:
                    if scale_factors is not None:
                        scale_factor = scale_factors[k]
                        v = (v - scale_factor[0]) / scale_factor[1]
                    vs.append(v)
                    ws.append(w)
        if np.isnan(np.array(vs)).any():
            return np.inf
        else:
            return np.nanprod(np.array(vs)*np.array(ws))

    def mining(self, labeled_obj, partial_obj, pred_data, ratio, cfg=dict()):
        """Mine boxes based on box_preds and scores.

        Args:
            partial_obj: COCO object including not fully labeled data (data in later cycles)
                # list of dict with image_id as key
                # "info": an image dict
                # "instances" (list): anno dicts
                # "cntbox" (int): #GT_boxes
            pred_data: list of predicted bboxes 
                "image_id": 
                "bbox" (list): xywh
                "category_id" (int): predicted label start from 0
                "score_rpn" (float): value nan means box predicted 
                "score_cls" (float): by T-model and fitered after thr
                "score_reg" (float): but not matched by any S-preds
            ratio: appended number of samples (<1 float or int)
            partial_data: list of dict with image_id as key
        """

        def xywh2xyxy(bbox):
            assert bbox.shape[1] == 4
            return np.concatenate((
                bbox[:, :1], 
                bbox[:, 1:2], 
                bbox[:, :1] + bbox[:, 2:3], 
                bbox[:, 1:2] + bbox[:, 3:], 
                ), axis=1)

        if ratio < 1:
            raise NotImplementedError("box-level AL do not support ratio-based budget \
                because u dont know the total #boxes")
        else:
            to_select_num = int(ratio)
        overlap_thresh = cfg.get('thresh', 0.25)  # un-finetuned
        assigner = MaxIoUAssigner(
            pos_iou_thr=overlap_thresh, 
            neg_iou_thr=overlap_thresh, 
            min_pos_iou=overlap_thresh, 
            gt_max_assign_all=False
            )
        # ank boxes and match w/ unannotated GTs
        weight = cfg.get("weight")
        box_scores_d = {k: [dic[k] for dic in pred_data] for k in pred_data[0].keys() if k.startswith("score_")}
        scale_factors = {k: [np.nanmin(v), np.nanmax(v) - np.nanmin(v)] for k, v in box_scores_d.items()}
        print(scale_factors)
        box_scores = np.array([self.value(pred_box, scale_factors=scale_factors, weight=weight) for pred_box in pred_data])
        box_args = np.argsort(box_scores)[::-1] 

        for box_idx in box_args:
            if to_select_num > 0:
                preds = pred_data[box_idx]
                img_id = preds["image_id"]
                img_gts = partial_obj.get_anns(imgIds=img_id, iscrowd=False)
                if len(img_gts) == 0:
                    continue
                img_gt_bboxes = np.zeros((len(img_gts), 4), dtype=np.float)
                img_gt_bboxes_labeled_prev = np.zeros(len(img_gts), dtype=np.bool)
                for i, gt in enumerate(img_gts):
                    img_gt_bboxes[i] = gt["bbox"]
                    img_gt_bboxes_labeled_prev[i] = gt["islabeled"]
                if np.isinf(box_scores[box_idx]):
                    continue
                else:
                    pred_bbox = xywh2xyxy(np.array([preds["bbox"]]))
                    img_gt_bboxes = xywh2xyxy(img_gt_bboxes)
                    tmp = assigner.assign(
                        torch.from_numpy(pred_bbox),
                        torch.from_numpy(img_gt_bboxes), 
                        gt_labels=None
                        ).gt_inds.numpy() - 1   # gt_inds is 1-based
                    if not (tmp >= 0).any():
                        continue
                
                for matched_ind in tmp:
                    if img_gt_bboxes_labeled_prev[matched_ind]:
                        continue
                    else:
                        ann_id = img_gts[matched_ind]["id"]
                        img_gts[matched_ind]["islabeled"] = True
                        num_unlabeled = partial_obj.set_anns_labeled(ann_id, return_unlabeled=True)
                        to_select_num -= 1

                        if num_unlabeled[0] == 0:
                            labeled_obj.add_image(partial_obj.load_imgs(img_id), [img_gts])
                            partial_obj.remove_image(img_id)

            else:
                break

        return labeled_obj, partial_obj