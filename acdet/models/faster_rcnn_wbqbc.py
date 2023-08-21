from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import FasterRCNN

@DETECTORS.register_module()
class FasterRCNNWB(FasterRCNN):
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
        super(FasterRCNNWB, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_test(self, imgs, img_metas, **kwargs):
        if kwargs.get("active_cycle") is not None:
            self.active_cycle = kwargs.pop("active_cycle")
            return self.simple_test_active(imgs, img_metas, **kwargs)
        else:
            return super().forward_test(imgs, img_metas, **kwargs)

    def simple_test_active(self, imgs, img_metas, proposals=None, rescale=False):
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

            multi_scale_boxes, multi_scale_labels = self.roi_head.simple_test_bboxes_active(
                x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)
            bbox_results = []
            for det_bboxes, det_labels in zip(multi_scale_boxes, multi_scale_labels):
                bbox_result = [
                    bbox2result(det_bboxes[i], det_labels[i],
                                self.roi_head.bbox_head.num_classes)
                    for i in range(len(det_bboxes))
                ]
                bbox_results.append(bbox_result)
            return bbox_results
        else:
            raise NotImplementedError

