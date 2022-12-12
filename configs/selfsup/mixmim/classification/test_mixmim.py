_base_ = [
    'mmcls::_base_/datasets/imagenet_bs64_swin_224.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmselfsup.MixMIMTransformer',
        arch='B',
        drop_rate=0.0,
        drop_path_rate=0.1),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

dataset_type = 'ImageNet'
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

bgr_mean = preprocess_cfg['mean'][::-1]

bgr_std = preprocess_cfg['std'][::-1]
dataset_type = 'ImageNet'
file_client_args = dict(backend='disk')
data_root = '/data/personal/nus-zwb/ImageNet/'
train_ann_file = '/home/nus-zwb/research/data/imagenet/meta/train.txt'
val_ann_file = '/home/nus-zwb/research/data/imagenet/meta/val.txt'

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * (8 * 128 / 256),
        model_type='mixmim',
        layer_decay_rate=0.7,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        custom_keys={
            '.ln': dict(decay_mult=0.0),  # do not decay on ln and bias
            '.bias': dict(decay_mult=0.0)
        }))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=100))
