import torch

from mmdet.core import bbox2roi, multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

@HEADS.register_module()
class ActiveStandardRoILearningLossHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(
            x[:self.bbox_roi_extractor.num_inputs], 
            bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        roi_splits = [res.bboxes.shape[0] for res in sampling_results]
        pos_inds_splits = [len(res.pos_inds) for res in sampling_results]
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets,
                                        roi_splits=roi_splits,
                                        pos_inds_splits=pos_inds_splits)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
