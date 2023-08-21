mmdet_config = "../../../../thirdparty/mmdetection/configs"
_base_ = f'{mmdet_config}/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
dataset_type = 'ActiveCocoDataset'
data_root = "data/detection/voc_converted/"

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    )
)

train_transforms = [
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
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=train_transforms,
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="labeled"),
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
            "tag",
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
    valid_len=16551,  # 47223 boxes, 40058 not crowded, 0712_trainval
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        labeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/initial.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=train_pipeline,
            classes=CLASSES),
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
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc0712_trainval.json',
        img_prefix=data_root + 'VOCdevkit/',
        pipeline=test_pipeline))
evaluation = dict(interval=2500, metric='bbox', save_best='bbox_mAP', classwise=True)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=12500)   # full is 25000, for 2*4 samples/iter
optimizer = dict(lr=0.01)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8500, 11500])
checkpoint_config = dict(by_epoch=False, interval=2500, max_keep_ckpts=5)

fp16 = dict(loss_scale="dynamic")