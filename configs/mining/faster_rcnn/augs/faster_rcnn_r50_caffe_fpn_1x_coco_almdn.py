_base_ = 'faster_rcnn_r50_caffe_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        type='ActiveStandardRoIALMDNHead',
        bbox_head=dict(
            type='ActiveShared2FCBBoxALMDNHead',
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(
                type='NLLLoss', loss_weight=1.0),
        ),
    )
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)