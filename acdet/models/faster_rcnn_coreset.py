from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import FasterRCNN
import torch

@DETECTORS.register_module()
class FasterRCNNCoreset(FasterRCNN):
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
        super(FasterRCNNCoreset, self).__init__(
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

            result = self.extract_feat(imgs[0])
            # use result.dim() to condition if use GAP
            result_list = []
            result = result[:self.roi_head.bbox_roi_extractor.num_inputs]
            for pyramid_feat in result:
                if pyramid_feat.dim() == 4:
                    pyramid_feat = pyramid_feat.mean(dim=(-1,-2))
                elif pyramid_feat.dim() == 3:
                    pyramid_feat = pyramid_feat.mean(dim=-1)
                result_list.append(pyramid_feat)
            return torch.cat(result_list, -1)
        else:
            raise NotImplementedError

