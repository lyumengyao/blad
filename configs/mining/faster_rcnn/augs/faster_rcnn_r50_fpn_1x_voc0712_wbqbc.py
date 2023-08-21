_base_ = 'faster_rcnn_r50_fpn_1x_voc0712.py'

model = dict(
    type='FasterRCNNWB',
    roi_head=dict(
        type='ActiveStandardRoIWBHead',
    )
)
