_base_ = 'faster_rcnn_r50_fpn_1x_voc0712.py'

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
    dict(type='DropLearningLossIterBasedHook', freeze_iter=10000)
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10000])