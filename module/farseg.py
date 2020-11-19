import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simplecv.interface import CVModule
from simplecv import registry
from simplecv.module import resnet
from simplecv.module import fpn
import math
from module.loss import softmax_focalloss
from module.loss import annealing_softmax_focalloss
from module.loss import cosine_annealing, poly_annealing, linear_annealing
import simplecv.module as scm


class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
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
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class AssymetricDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(AssymetricDecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


@registry.MODEL.register('FarSeg')
class FarSeg(CVModule):
    def __init__(self, config):
        super(FarSeg, self).__init__(config)
        self.register_buffer('buffer_step', torch.zeros((), dtype=torch.float32))

        self.en = resnet.ResNetEncoder(self.config.resnet_encoder)
        self.fpn = fpn.FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if 'scene_relation' in self.config:
            print('scene_relation: on')
            self.gap = scm.GlobalAvgPool2D()
            self.sr = SceneRelation(**self.config.scene_relation)

        if 'softmax_focalloss' in self.config:
            print('loss type: softmax_focalloss')

        if 'cosineannealing_softmax_focalloss' in self.config:
            print('loss type: cosineannealing_softmax_focalloss')

        if 'annealing_softmax_focalloss' in self.config:
            print('loss type: {}'.format(self.config.annealing_softmax_focalloss.annealing_type))

    def forward(self, x, y=None):
        feat_list = self.en(x)
        fpn_feat_list = self.fpn(feat_list)
        if 'scene_relation' in self.config:
            c5 = feat_list[-1]
            c6 = self.gap(c5)
            refined_fpn_feat_list = self.sr(c6, fpn_feat_list)
        else:
            refined_fpn_feat_list = fpn_feat_list

        final_feat = self.decoder(refined_fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)

        if self.training:
            cls_true = y['cls']
            loss_dict = dict()
            self.buffer_step += 1
            cls_loss_v = self.config.loss.cls_weight * self.cls_loss(cls_pred, cls_true)
            loss_dict['cls_loss'] = cls_loss_v

            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        return cls_pred.softmax(dim=1)

    def cls_loss(self, y_pred, y_true):
        if 'softmax_focalloss' in self.config:
            return softmax_focalloss(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index,
                                     gamma=self.config.softmax_focalloss.gamma,
                                     normalize=self.config.softmax_focalloss.normalize)
        elif 'annealing_softmax_focalloss' in self.config:
            func_dict = dict(cosine=cosine_annealing,
                             poly=poly_annealing,
                             linear=linear_annealing)
            return annealing_softmax_focalloss(y_pred, y_true.long(),
                                               self.buffer_step.item(),
                                               self.config.annealing_softmax_focalloss.max_step,
                                               self.config.loss.ignore_index,
                                               self.config.annealing_softmax_focalloss.gamma,
                                               func_dict[self.config.annealing_softmax_focalloss.annealing_type])
        return F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

    def set_defalut_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
                norm_layer=nn.BatchNorm2d,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=fpn.default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=16,
            loss=dict(
                cls_weight=1.0,
                ignore_index=255,
            )
        ))
