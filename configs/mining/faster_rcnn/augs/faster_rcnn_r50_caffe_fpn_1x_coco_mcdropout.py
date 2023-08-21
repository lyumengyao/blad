_base_ = 'faster_rcnn_r50_caffe_fpn_1x_coco.py'

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
