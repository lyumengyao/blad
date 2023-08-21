_base_ = 'faster_rcnn_r50_fpn_1x_voc0712.py'

model = dict(
    type='FasterRCNN',
    backbone=dict(plugins=[
        dict(
            cfg=dict(
                type='MCDropout',
                p=0.1),
            stages=(False, False, True, True),
            position='after_conv2')
    ])
)
