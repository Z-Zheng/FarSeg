import ever as er
from ever.module import PyramidPoolModule, ConvBlock, ConvUpsampling
import torch.nn as nn
import torch
from ever.module import FPN, AssymetricDecoder
import torch.nn.functional as F
from ever.module import ResNetEncoder
from module.comm import MultiSegmentation


class FSRelation(nn.Module):
    def __init__(self,
                 scene_embedding_channels,
                 in_channels_list,
                 out_channels,
                 scale_aware_proj=False,
                 ):
        super(FSRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(scene_embedding_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(True),
                ) for _ in range(len(in_channels_list))]
            )
            self.project = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Dropout2d(p=0.1)
                ) for _ in range(len(in_channels_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_embedding_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(True),
            )
            self.project = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Dropout2d(p=0.1)
            )

        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        # [N, C, H, W]
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            # [N, C, 1, 1]
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [torch.cat([r * p, o], dim=1) for r, p, o in zip(relations, p_feats, features)]

        if self.scale_aware_proj:
            ffeats = [op(x) for op, x in zip(self.project, refined_feats)]
        else:
            ffeats = [self.project(x) for x in refined_feats]

        return ffeats


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 ):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, inner_channels, 1, bias=False)
        self.conv2 = ConvBlock(inner_channels, inner_channels, 3, 1, 1, bias=False)
        self.conv3 = ConvBlock(inner_channels, out_channels, 1, bias=False, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity
        out = self.relu(out)
        return out


class ResolutionAlign(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.project = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1, bias=False),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, features):
        resized = [features[0]]
        dst_h = resized[0].size(2)
        dst_w = resized[0].size(3)
        for feat in features[1:]:
            if feat.size(2) == dst_h and feat.size(3) == dst_w:
                resized.append(feat)
            else:
                resized.append(F.interpolate(feat,
                                             size=(dst_h,
                                                   dst_w),
                                             mode='bilinear',
                                             align_corners=True
                                             ))
        return self.project(torch.cat(resized, dim=1))


class Decoder(AssymetricDecoder):
    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        if self.cls_cfg:
            logit = self.dropout(out_feat)
            logit = self.classifier(logit)
        return logit, out_feat


class ParallelDecoder(nn.Module):
    def __init__(self, obj_cfg, seg_cfg):
        super().__init__()
        self.obj_decoder = Decoder(**obj_cfg)
        self.seg_decoder = Decoder(**seg_cfg)

    def forward(self, features):
        if self.training:
            obj_logit, _ = self.obj_decoder(features)
        else:
            obj_logit = None
        seg_logit, _ = self.seg_decoder(features)
        return obj_logit, seg_logit


class SegObjCascadeDecoder(nn.Module):
    def __init__(self, obj_cfg, seg_cfg):
        super().__init__()
        self.seg_decoder = Decoder(**seg_cfg)
        if 'use_obj_logit' in obj_cfg:
            self.use_obj_logit = obj_cfg.pop('use_obj_logit')
        else:
            self.use_obj_logit = False
        self.obj_decoder = Decoder(**obj_cfg)

        self.conv = ConvBlock(seg_cfg.out_channels, obj_cfg.in_channels, 1, bias=False)

    def forward(self, features):
        seg_logit, seg_feature = self.seg_decoder(features)
        if self.training or self.use_obj_logit:
            obj_logit, _ = self.obj_decoder([self.conv(seg_feature)] + features)
        else:
            obj_logit = None
        return obj_logit, seg_logit


class ObjSegCascadeDecoder(nn.Module):
    def __init__(self, obj_cfg, seg_cfg):
        super().__init__()
        self.obj_decoder = Decoder(**obj_cfg)
        self.seg_decoder = Decoder(**seg_cfg)
        self.conv = ConvBlock(obj_cfg.out_channels, seg_cfg.in_channels, 1, bias=False)

    def forward(self, features):
        obj_logit, obj_feature = self.obj_decoder(features)
        seg_logit, _ = self.seg_decoder([self.conv(obj_feature)] + features)
        return obj_logit, seg_logit


@er.registry.MODEL.register()
class FarSegPP(er.ERModule, MultiSegmentation):
    def __init__(self, config):
        super().__init__(config)
        if self.config.backbone.type == 'resnet':
            self.en = ResNetEncoder(self.config.backbone)
        elif self.config.backbone.type == 'mit':
            from module.mit import MiTEncoder
            self.en = MiTEncoder(self.config.backbone)

        del self.de
        self.ppm = PyramidPoolModule(**self.config.ppm)
        self.fsr = FSRelation(**self.config.fs_relation)

        self.fpn = FPN(**self.config.fpn)
        if self.config.decoder_arch == 'ParallelDecoder':
            self.decoder = ParallelDecoder(
                self.config.obj_asy_decoder,
                self.config.asy_decoder
            )
        elif self.config.decoder_arch == 'SegObjCascadeDecoder':
            self.decoder = SegObjCascadeDecoder(
                self.config.obj_asy_decoder,
                self.config.asy_decoder
            )
        elif self.config.decoder_arch == 'ObjSegCascadeDecoder':
            self.decoder = ObjSegCascadeDecoder(
                self.config.obj_asy_decoder,
                self.config.asy_decoder
            )
        self.register_buffer('buffer_step', torch.zeros((), dtype=torch.float32))

    def forward(self, x, y=None):
        feature_list = self.en(x)
        last_feat = feature_list[-1]
        # ppm
        feature_list[-1] = self.ppm(feature_list[-1])
        # fpn
        fpn_feature_list = self.fpn(feature_list)
        # fsr
        scene_embedding = F.adaptive_avg_pool2d(last_feat, 1)
        refined_fpn_feature_list = self.fsr(scene_embedding, fpn_feature_list)
        # decode
        obj_logit, seg_logit = self.decoder(refined_fpn_feature_list)

        if self.training:
            loss_dict = dict()
            gt_seg = y['cls']
            gt_binary_seg = torch.where(((gt_seg > 0) & (gt_seg != self.config.loss.objectness.ignore_index)),
                                        torch.ones_like(gt_seg),
                                        gt_seg).float()
            loss_dict.update(self.loss(gt_binary_seg, obj_logit, self.config.loss.objectness))
            self.buffer_step += 1.
            loss_dict.update(self.loss(gt_seg, seg_logit, self.config.loss.semantic, buffer_step=self.buffer_step))

            return loss_dict
        if hasattr(self.decoder, 'use_obj_logit') and self.decoder.use_obj_logit:
            return obj_logit.sigmoid()
        return seg_logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                type='resnet',
                in_channels=3,
                resnet_type='resnet50_v1c',
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
            ),
            ppm=dict(
                in_channels=2048,
                pool_channels=512,
                out_channels=512,
                bins=(1, 2, 3, 6),
                bottleneck_conv='1x1',
                dropout=0.1
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 512),
                out_channels=256,
            ),
            fs_relation=dict(
                scene_embedding_channels=2048,
                in_channels_list=(256, 256, 256, 256),
                out_channels=256,
                scale_aware_proj=True
            ),
            decoder_arch='ObjSegCascadeDecoder',
            obj_asy_decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
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
                in_feat_output_strides=(4, 4, 8, 16, 32),
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
                    ignore_index=255,
                    prefix='obj_'
                ),
                semantic=dict(
                    log_objectness_iou=dict(),
                    ignore_index=255,
                )
            ),
        ))

    def log_info(self):
        return dict(cfg=self.config)

    def custom_param_groups(self):
        if self.config.backbone.type == 'mit':
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        elif self.config.backbone.type == 'swin':
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for i, p in self.named_parameters():
                if 'norm' in i:
                    param_groups[0]['params'].append(p)
                elif 'relative_position_bias_table' in i:
                    param_groups[0]['params'].append(p)
                elif 'absolute_pos_embed' in i:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return super().custom_param_groups()
