_base_ = 'faster_rcnn_r50_caffe_fpn_1x_coco.py'

dataset_wrapper = 'SemiDataset'
dataset_type = 'ActiveCocoDataset'
data_root = "data/detection/coco/"
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    type='FasterRCNNMIAOD',
    roi_head=dict(
        type='ActiveStandardRoIMIAODHead',
        bbox_head=dict(
            type='ActiveShared2FCBBoxMIAODHead',
        ),
    )
)

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

labeled_train_pipeline = [
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

unlabeled_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="PseudoSamples", with_bbox=True), # fake GTs
    dict(
        type="Sequential",
        transforms=train_transforms,
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unlabeled"),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_wrapper,
        labeled=dict(
            pipeline=labeled_train_pipeline,
        ),
        unlabeled=dict(
            pipeline=unlabeled_train_pipeline,
        ),
    ),
    sampler=dict(
        train=dict(
            type="MultiBalanceSampler",
            sample_ratio=[1, 1],
            by_prob=False,
            at_least_one=False,
            epoch_length=7330, 
        )
    )
)

head_cls_1 = ['roi_head.bbox_head.fc_cls_1.weight', 'roi_head.bbox_head.fc_cls_1.bias']
head_cls_2 = ['roi_head.bbox_head.fc_cls_2.weight', 'roi_head.bbox_head.fc_cls_2.bias']

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='SwitchStageIterBasedHook', 
        stages=[
                dict(
                    sample_ratio=[1, 0],
                    loss_type=0,
                    epochs=5,
                    validate=False
                )
            ] +  # labeled training at epoch 0
            [
                dict(
                    sample_ratio=[1, 1],
                    loss_type=1,
                    epochs=1,
                    validate=False,
                    params_freeze=head_cls_1+head_cls_2,
                ),  # Re-weighting and Minimizing Instance Uncertainty
                dict(
                    sample_ratio=[1, 1],
                    loss_type=2,
                    epochs=1,
                    validate=False,
                    params_learn=head_cls_1+head_cls_2,
                ),
                dict(
                    sample_ratio=[1, 0],
                    loss_type=0,
                    epochs=5,
                    validate=True
                ),
            ] * 2
    )
]


optimizer = dict(lr=0.005, momentum=0.9, weight_decay=0.0001)
fp16 = dict(loss_scale="dynamic")

log_config = dict(interval=100)

repeat_times = 32