mmdet_config = "../../../thirdparty/mmdetection/configs"
_base_ = f'{mmdet_config}/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
dataset_type = 'ActiveCocoDataset'
data_root = "data/detection/voc_converted/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    valid_len=16551,  # 47223 boxes, 0712_trainval
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        initial=dict(
            type='RepeatDataset',
            times=3,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root + 'annotations/initial.json',
                img_prefix=data_root + 'VOCdevkit/',
                pipeline=train_pipeline,
                classes=CLASSES)),
        unlabeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/unlabeled.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=train_pipeline,
            classes=CLASSES),
        full=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/voc0712_trainval.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=train_pipeline,
            classes=CLASSES)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc07_test.json',
        img_prefix=data_root + 'VOCdevkit/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc0712_trainval.json',
        img_prefix=data_root + 'VOCdevkit/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', classwise=True)