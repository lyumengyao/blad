import torch

from mmdet.core import bbox2roi, multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

@HEADS.register_module()
class ActiveStandardRoIWBHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def _bbox_forward_test(self, x, rois):
        """Box head forward function used in active learning."""
        # TODO: a more flexible way to decide which feature maps to use
        cls_scores, bbox_preds, bbox_feats = [], [], []
        for i in range(self.bbox_roi_extractor.num_inputs):
            bbox_feat = self.bbox_roi_extractor([x[i]], rois)
            if self.with_shared_head:
                bbox_feat = self.shared_head(bbox_feat)
            cls_score, bbox_pred = self.bbox_head(bbox_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            bbox_feats.append(bbox_feat)

        return cls_scores, bbox_preds, bbox_feats

    def simple_test_bboxes_active(self,
                                  x,
                                  img_metas,
                                  proposals,
                                  rcnn_test_cfg,
                                  rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward_test(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)

        multi_scale_boxes, multi_scale_labels = [], []
        for cls_score, bbox_pred, _ in zip(*bbox_results):
            # split batch bbox prediction back to each image
            cls_score = cls_score.split(num_proposals_per_img, 0)

            # some detector with_reg is False, bbox_pred will be None
            if bbox_pred is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred, torch.Tensor):
                    bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                else:
                    bbox_pred = self.bbox_head.bbox_pred_split(
                        bbox_pred, num_proposals_per_img)
            else:
                bbox_pred = (None, ) * len(proposals)

            # apply bbox post-processing to each image individually
            det_bboxes = []
            det_labels = []
            for i in range(len(proposals)):
                if rois[i].shape[0] == 0:
                    # There is no proposal in the single image
                    det_bbox = rois[i].new_zeros(0, 5)
                    det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                    if rcnn_test_cfg is None:
                        det_bbox = det_bbox[:, :4]
                        det_label = rois[i].new_zeros(
                            (0, self.bbox_head.fc_cls.out_features))
                else:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            multi_scale_boxes.append(det_bboxes)
            multi_scale_labels.append(det_labels)
        return multi_scale_boxes, multi_scale_labels
