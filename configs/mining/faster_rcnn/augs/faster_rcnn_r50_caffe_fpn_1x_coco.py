mmdet_config = "../../../../thirdparty/mmdetection/configs"
_base_ = f'{mmdet_config}/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py'

dataset_type = 'ActiveCocoDataset'
data_root = "data/detection/coco/"

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_transforms = [
    dict(
        type="RandResize",
        img_scale=[(1333, 400), (1333, 1200)],
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
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    valid_len=117266,   # 849949 not iscrowded boxes
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        labeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/initial.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline),
        unlabeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/unlabeled.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline),
        full=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=17600, metric='bbox', save_best='bbox_mAP', classwise=True)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=88000)   # full is 175920, for 2*4 samples/iter
lr_config = dict(step=[59000, 81000])
checkpoint_config = dict(by_epoch=False, interval=17600, max_keep_ckpts=10)

fp16 = dict(loss_scale="dynamic")