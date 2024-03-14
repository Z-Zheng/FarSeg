from ever.preprocess.albu import ToTensor, ConstantPad, RandomDiscreteScale
import albumentations as A

data = dict(
    train=dict(
        type='OfflinePatchiSAIDDataLoader',
        params=dict(
            image_dir='./.cache_iSAID_data/image',
            mask_dir='./.cache_iSAID_data/mask',
            transforms=A.Compose([
                RandomDiscreteScale([1.25, 1.5, 1.75, 2.0], p=0.5),
                A.OneOf([
                    A.HorizontalFlip(True),
                    A.VerticalFlip(True),
                    A.RandomRotate90(True)
                ], p=0.75),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225), max_pixel_value=255),
                ConstantPad(896, 896, 0, 255),
                A.RandomCrop(896, 896),
                ToTensor()
            ]),
            batch_size=4,
            num_workers=4,
            training=True
        ),
    )
)
max_iters = 120000
train = dict(
    forward_times=1,
    num_iters=max_iters,
    eval_per_epoch=False,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=False,
    log_interval_step=100,
    save_ckpt_interval_epoch=999,
    eval_interval_epoch=999,
)
test = dict(
    multimodal=False
)
optimizer = dict(
    type='adamw',
    params=dict(
        weight_decay=0.01
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.00006,
        power=0.9,
        max_iters=max_iters,
    )
)

config = dict(
    model=dict(
        type='FarSegPP',
        params=dict(
            backbone=dict(
                type='mit',
                name='mit_b2',
                pretrained='./pretrained/mit_b2.pth',
                drop_path_rate=0.1
            ),
            ppm=dict(
                in_channels=512,
                pool_channels=128,
                out_channels=128,
                bins=(1, 2, 3, 6),
                bottleneck_conv='1x1',
                dropout=0.1
            ),
            fpn=dict(
                in_channels_list=(64, 128, 320, 128),
                out_channels=256,
            ),
            fs_relation=dict(
                scene_embedding_channels=512,
                in_channels_list=(256, 256, 256, 256),
                out_channels=256,
                scale_aware_proj=True
            ),
            decoder_arch='SegObjCascadeDecoder',
            obj_asy_decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 4, 8, 16, 32),
                out_feat_output_stride=4,
                classifier_config=dict(
                    scale_factor=4.0,
                    num_classes=1,
                    kernel_size=3
                )
            ),
            asy_decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                classifier_config=dict(
                    scale_factor=4.0,
                    num_classes=16,
                    kernel_size=3
                )
            ),
            loss=dict(
                objectness=dict(
                    log_objectness_iou_sigmoid=dict(),
                    dice=dict(),
                    bce=dict(),
                    ignore_index=255,
                    prefix='obj_'
                ),
                semantic=dict(
                    annealing_softmax_focal=dict(normalize=True, t_max=0),
                    log_objectness_iou=dict(),
                    ignore_index=255,
                )
            ),
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test,
)
