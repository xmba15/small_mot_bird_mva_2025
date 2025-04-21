default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        save_best="auto",
        type="CheckpointHook",
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(
        type="mmdet.DetVisualizationHook",
        draw=True,
        interval=50,
        show=False,
        score_thr=0.3,
    ),
)

default_scope = "mmyolo"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

log_level = "INFO"
log_processor = dict(
    by_epoch=True,
    type="LogProcessor",
    window_size=50,
)

num_classes = 1
max_epochs = 10
deepen_factor = 1.0
img_scale = (800, 800)
train_batch_size = 4
accumulative_counts = 2
lr_rate = 5.0e-4
dataset_type = "mmdet.CocoDataset"

model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type="SiLU"),
        arch="P5",
        deepen_factor=deepen_factor,
        last_stage_out_channels=512,
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        type="YOLOv8CSPDarknet",
        widen_factor=deepen_factor,
    ),
    bbox_head=dict(
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        head_module=dict(
            act_cfg=dict(inplace=True, type="SiLU"),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
            num_classes=num_classes,
            reg_max=16,
            type="YOLOv8HeadModule",
            widen_factor=deepen_factor,
        ),
        loss_bbox=dict(
            bbox_format="xyxy",
            iou_mode="ciou",
            loss_weight=7.5,
            reduction="sum",
            return_iou=False,
            type="IoULoss",
        ),
        loss_cls=dict(
            loss_weight=0.5,
            reduction="none",
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
        ),
        loss_dfl=dict(
            loss_weight=0.375, reduction="mean", type="mmdet.DistributionFocalLoss"
        ),
        prior_generator=dict(
            offset=0.5,
            strides=[
                8,
                16,
                32,
            ],
            type="mmdet.MlvlPointGenerator",
        ),
        type="YOLOv8Head",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="mmdet.DetDataPreprocessor",
    ),
    neck=dict(
        act_cfg=dict(inplace=True, type="SiLU"),
        deepen_factor=deepen_factor,
        in_channels=[
            256,
            512,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            512,
        ],
        type="YOLOv8PAFPN",
        widen_factor=deepen_factor,
    ),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type="nms"),
        nms_pre=30000,
        score_thr=0.001,
    ),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=num_classes,
            topk=10,
            type="BatchTaskAlignedAssigner",
            use_ciou=True,
        )
    ),
    type="YOLODetector",
)

train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type="mmdet.PhotoMetricDistortion",
    ),
    dict(
        crop_size=img_scale,
        mean=[0, 0, 0],
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        std=[
            1,
            1,
            1,
        ],
        test_pad_mode=None,
        to_rgb=True,
        type="mmdet.RandomCenterCropPad",
    ),
    dict(
        keep_ratio=True,
        scale=img_scale,
        type="Resize",
    ),
    dict(prob=0.5, type="mmdet.RandomFlip"),
    dict(type="PackDetInputs"),
]

data_root = "./data/processed_smot4sb/fold_0/"
train_cfg = dict(
    max_epochs=max_epochs,
    type="EpochBasedTrainLoop",
    val_interval=1,
)
train_dataloader = dict(
    batch_sampler=dict(type="mmdet.AspectRatioBatchSampler"),
    batch_size=train_batch_size,
    dataset=dict(
        ann_file="annotations/train.json",
        metainfo=dict(classes=("bird",)),
        backend_args=None,
        data_prefix=dict(img="train/"),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        type=dataset_type,
    ),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)

val_cfg = dict(
    type="ValLoop",
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/val.json",
        metainfo=dict(classes=("bird",)),
        backend_args=None,
        data_prefix=dict(img="val/"),
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, to_float32=True, type="LoadImageFromFile"),
            dict(
                border=None,
                mean=[0, 0, 0],
                ratios=None,
                std=[1, 1, 1],
                test_mode=True,
                test_pad_add_pix=1,
                test_pad_mode=[
                    "logical_or",
                    31,
                ],
                to_rgb=True,
                type="mmdet.RandomCenterCropPad",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "border",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type=dataset_type,
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

val_evaluator = dict(
    type="mmdet.CocoMetric",
    ann_file=data_root + "annotations/val.json",
    metric="bbox",
    format_only=False,
    backend_args=None,
)

optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        lr=lr_rate,
        type="AdamW",
        weight_decay=0.01,
    ),
    type="OptimWrapper",
)
param_scheduler = [
    dict(
        T_max=max_epochs,
        by_epoch=True,
        eta_min=1.0e-05,
        type="CosineAnnealingLR",
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(
    name="visualizer",
    type="mmdet.DetLocalVisualizer",
    vis_backends=vis_backends,
)
