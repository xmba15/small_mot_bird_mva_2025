default_hooks = dict(
    checkpoint=dict(interval=1, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(
        type="DetVisualizationHook",
        draw=True,
        interval=50,
        show=False,
        score_thr=0.3,
    ),
)

default_scope = "mmdet"
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

max_epochs = 20

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(checkpoint="torchvision://resnet18", type="Pretrained"),
        norm_cfg=dict(type="BN"),
        norm_eval=False,
        type="ResNet",
    ),
    bbox_head=dict(
        feat_channels=64,
        in_channels=64,
        loss_center_heatmap=dict(loss_weight=1.0, type="GaussianFocalLoss"),
        loss_offset=dict(loss_weight=1.0, type="L1Loss"),
        loss_wh=dict(loss_weight=0.1, type="L1Loss"),
        num_classes=1,
        type="CenterNetHead",
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
        type="DetDataPreprocessor",
    ),
    neck=dict(
        in_channels=512,
        num_deconv_filters=(
            256,
            128,
            64,
        ),
        num_deconv_kernels=(
            4,
            4,
            4,
        ),
        type="CTResNetNeck",
        use_dcn=False,
    ),
    test_cfg=dict(local_maximum_kernel=3, max_per_img=100, topk=100),
    train_cfg=None,
    type="CenterNet",
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
        type="PhotoMetricDistortion",
    ),
    dict(
        crop_size=(800, 800),
        mean=[0, 0, 0],
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        std=[
            1,
            1,
            1,
        ],
        test_pad_mode=None,
        to_rgb=True,
        type="RandomCenterCropPad",
    ),
    dict(
        keep_ratio=True,
        scale=(800, 800),
        type="Resize",
    ),
    dict(prob=0.5, type="RandomFlip"),
    dict(type="PackDetInputs"),
]

data_root = "./data/processed_smot4sb/fold_0/"
train_cfg = dict(
    max_epochs=max_epochs,
    type="EpochBasedTrainLoop",
    val_interval=1,
)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=16,
    dataset=dict(
        ann_file="annotations/train.json",
        metainfo=dict(classes=("bird",)),
        backend_args=None,
        data_prefix=dict(img="train/"),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        type="CocoDataset",
    ),
    num_workers=4,
    persistent_workers=True,
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
                type="RandomCenterCropPad",
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
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/val.json",
    metric="bbox",
    format_only=False,
    backend_args=None,
)

optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type="AdamW", weight_decay=0.01),
    type="OptimWrapper",
)
param_scheduler = [
    dict(
        T_max=max_epochs,
        by_epoch=True,
        eta_min=1e-05,
        type="CosineAnnealingLR",
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
)
