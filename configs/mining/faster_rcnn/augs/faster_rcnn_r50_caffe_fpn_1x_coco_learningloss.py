_base_ = 'faster_rcnn_r50_caffe_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        type='ActiveStandardRoILearningLossHead',
        bbox_head=dict(
            type='ActiveShared2FCBBoxLearningLossHead'
        ),
    )
)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='DropLearningLossIterBasedHook', freeze_iter=70400)
]