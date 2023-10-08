_base_ = [
    '../_base_/datasets/cityscapes_.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(0,1, 2),
        strides=(1, 2, 2),
        arch_settings = [[1, 24, 1], [6, 32, 2], [6, 48, 3]],
        dilations=(1, 1, 1),
        ),

    decode_head=dict(
        type='BaHeadBase',
        layer0=[3,2,3,[24,32,48],24,3,False],
        layer1=[3,2,3,[48,48,24],24,3,'se_a'],
        layer2=[3,2,3,[48,48,24],24,3,'se_a'],
        layer3=[2,1,3,[48,48,24],24,3,'se_a'],
        layer4=[1,0,3,[48,48,24],24,3,'se_a'],
        base_channel=24,
        channels=6,
        dilation=[2,3,5,7,3],
        dilation_list=[2],
        ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

evaluation = dict(interval=2000,metric='mIoU', pre_eval=True)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,)