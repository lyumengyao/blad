_base_ = './faster_rcnn_r50_fpn_1x_voc0712.py'
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
           
dataset_type = 'ActiveCocoDataset'
data_root = "data/detection/voc_converted/"

model = dict(
    type='FasterRCNNCoreset'
)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

coreset_pipeline = [
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
    coreset=dict(
        labeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/labeled.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=coreset_pipeline,
            classes=CLASSES),
        unlabeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/unlabeled.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=coreset_pipeline,
            classes=CLASSES),
        metric='cosine',
))
