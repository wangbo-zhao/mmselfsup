_base_ = '../../../benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'  # noqa: E501

# optimizer
optimizer = dict(type='mmselfsup.LARS', lr=3.2, weight_decay=0.0, momentum=0.9)
optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=optimizer, _delete_=True)



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


file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=2048, dataset=dict(pipeline=train_pipeline, ann_file=train_ann_file), drop_last=True)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, ann_file=test_ann_file), drop_last=False)
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, ann_file=test_ann_file), drop_last=False)



# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=100)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        eta_min=0.0,
        convert_to_iter_based=True)
]
