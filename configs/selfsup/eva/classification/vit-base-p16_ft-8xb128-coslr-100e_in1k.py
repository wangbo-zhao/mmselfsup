_base_ = '../../../benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'  # noqa: E501

# model settings
model = dict(
    head=dict(init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02)
                        ]),  # MAE sets std to 2e-5
)

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(lr=4e-4),  # layer-wise lr decay factor
)


dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))
train_ann_file = "/mnt/petrelfs/zhaowangbo/research/2022ICLR/data/imagenet/meta/train.txt"
test_ann_file = "/mnt/petrelfs/zhaowangbo/research/2022ICLR/data/imagenet/meta/val.txt"

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
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline, ann_file=train_ann_file))
val_dataloader = dict(batch_size=128, dataset=dict(pipeline=test_pipeline, ann_file=test_ann_file))
test_dataloader = val_dataloader
