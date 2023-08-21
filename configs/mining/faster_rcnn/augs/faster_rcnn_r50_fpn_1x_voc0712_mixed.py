mmdet_config = "../../../../thirdparty/mmdetection/configs"
_base_ = f'{mmdet_config}/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
dataset_wrapper = 'MixedDataset'
dataset_type = 'ActiveCocoDataset'
data_root = "data/detection/voc_converted/"
# for pytorch-based
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# for caffe-based
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
strong_transforms = [
            dict(
                type="RandResize",
                img_scale=[(1000, 300), (1000, 900)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
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
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ]
weak_transforms = [
            dict(
                type="RandResize",
                img_scale=[(1000, 300), (1000, 900)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ]

# pipelines for normal training
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
            # dict(type="ExtraAttrs", tag="test"),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                    keys=['img'],
                    meta_keys=(
                        "filename",
                        "ori_shape",
                        "img_shape",
                        "img_norm_cfg",
                        "pad_shape",
                        "scale_factor",
                        # "tag",
                    )
                ),
        ])
]

# pipelines for partial training
partial_strong_pipeline = [
    dict(
        type="Sequential",
        transforms=strong_transforms,
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="partial_strong"),
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
            "transform_matrix",
        ),
    ),
]
partial_weak_pipeline = [
    dict(
        type="Sequential",
        transforms=weak_transforms,
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="partial_weak"),
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
            "transform_matrix",
        ),
    ),
]
partial_pipeline = [
    dict(type="LoadImageFromFile"),
    # load GTs, to debug query, otherwise uncomment the next line
    dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", 
        partial_strong=partial_strong_pipeline, 
        partial_weak=partial_weak_pipeline,
    ),
]

# pipelines for unlabel training
unlabel_strong_pipeline = [
    dict(
        type="Sequential",
        transforms=strong_transforms,
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unlabel_strong"),
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
            "transform_matrix",
        ),
    ),
]
unlabel_weak_pipeline = [
    dict(
        type="Sequential",
        transforms=weak_transforms,
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unlabel_weak"),
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
            "transform_matrix",
        ),
    ),
]
unlabel_pipeline = [
    dict(type="LoadImageFromFile"),
    # load GTs, to debug query, otherwise uncomment the next line
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", 
        unlabel_strong=unlabel_strong_pipeline, 
        unlabel_weak=unlabel_weak_pipeline,
    ),
]
# pipelines for active evaluation
unlabel_pipeline_test = [
    dict(type="LoadImageFromFile"),
    # load GTs, to debug query, otherwise uncomment the next line
    dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranchTest", 
        unlabel_strong=unlabel_strong_pipeline, 
        unlabel_weak=unlabel_weak_pipeline,
        aug_times=10,
        # test=test_pipeline
    ),
]

data = dict(
    valid_len=16551,  # 47223 boxes, 0712_trainval
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        _delete_=True,
        type=dataset_wrapper,
        labeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/labeled.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=labeled_train_pipeline,
            classes=CLASSES),
        mixed=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/mixed.json',
            img_prefix=data_root + 'VOCdevkit/',
            # pipeline=unlabel_pipeline,
            classes=CLASSES),
        partial=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/partial.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=partial_pipeline,
            classes=CLASSES),
        unlabeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/unlabeled.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=unlabel_pipeline,
            classes=CLASSES),
        full=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/voc0712_trainval.json',
            img_prefix=data_root + 'VOCdevkit/',
            pipeline=labeled_train_pipeline,
            classes=CLASSES)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc07_test.json',
        img_prefix=data_root + 'VOCdevkit/',
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc07_test.json',
        img_prefix=data_root + 'VOCdevkit/',
        pipeline=test_pipeline,
        classes=CLASSES),
    pool=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc0712_trainval.json',
        img_prefix=data_root + 'VOCdevkit/',
        pipeline=unlabel_pipeline_test,
        classes=CLASSES),
    sampler=dict(
        train=dict(
            type="MultiBalanceSampler",
            sample_ratio=[1, 1, 4],
            by_prob=True,
            at_least_one=True,
            epoch_length=2070,
        )
    ),
)

mixed_wrapper = dict(
    type="MixedTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        partial_boxmerge_iou_thresh=0.4,
        partial_weight=1.0
    ),
    test_cfg=dict(inference_on="student"),
)

semi_wrapper = mixed_wrapper

active_wrapper = dict(
    type="MixedTeacher",
    model="${model}",
    test_cfg=dict(
        inference_on="student",
        unlabel_weight=dict(
            loss_cls=1.0,
            loss_bbox=1.0
        ),
        aug_cfg=dict(
            use_teacher_proposal=False,
            pseudo_label_initial_score_thr=0.5,
            rev_thr=False,
            rpn_pseudo_threshold=0.5,
            cls_pseudo_threshold=0.5,
            reg_pseudo_threshold=0.1,
            jitter_times=10,
            jitter_scale=0.06,
            min_pseduo_box_size=0,
            unsup_weight=4.0,
        ),
    )
)
# TOOD: mixed-wrapper and partial-wrapper

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]

evaluation = dict(type="SubModulesDistEvalHook", interval=2500, metric='bbox', save_best='bbox_mAP', classwise=True)
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

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False)
    ],
)