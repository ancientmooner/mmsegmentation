# model link: https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/dnl_debug/dnl_r101-d8_769x769_benchmark_cityscapes.pth
_base_ = './dnl_r50-d8_769x769_80k_cityscapes.py'
model = dict(
    pretrained=None,
    backbone=dict(depth=101, stem_channels=128),
    decode_head=dict(channels=512),
    auxiliary_head=dict(channels=512))
# dataset settings
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1, 1, 1], to_rgb=False)
crop_size = (769, 769)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
