mmdet_config = "../../../thirdparty/mmdetection/configs"
_base_ = f'{mmdet_config}/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    )
)

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# dataset settings
dataset_type = 'CocoDataset'
data_root = "data/detection/voc_converted/"

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 300), (1000, 900)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
        ),
    ),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True, 
        type=dataset_type,
        img_prefix='data/VOCdevkit',
        classes=CLASSES,
        # type='RepeatDataset',
        # times=3,
        # dataset=dict(
            ann_file=data_root + 'annotations/voc0712_trainval.json',
            pipeline=train_pipeline
            # )
        ),
    val=dict(
        ann_file=data_root + 'annotations/voc07_test.json',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'annotations/voc07_test.json',
        pipeline=test_pipeline))
# evaluation = dict(interval=2, metric='bbox')
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=25000)
lr_config = dict(step=[17000, 23000])
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

checkpoint_config = dict(interval=5000)
evaluation = dict(interval=5000, metric='bbox')