# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcls.models.backbones.vision_transformer import VisionTransformer
from mmseg.models.utils import PatchEmbed, nlc_to_nchw, nchw_to_nlc

from mmseg.models.backbones.mit import MixFFN
from ex_attention import EX_Module
from mmseg.models.utils.inverted_residual import InvertedResidual
from mmseg.models.utils.PSA import PSA_p
from mmseg.models.utils.se_layer import SELayer

from mmengine.visualization import Visualizer
def draw_feat(feature, visualizer, ori_img=None, draw_feat=False):
    if draw_feat is True:
        drawn_img = visualizer.draw_featmap(feature.squeeze(0),
                                            overlaid_image=ori_img,
                                            channel_reduction='select_max',
                                            resize_shape=(256, 256))
        visualizer.show(drawn_img, wait_time=1)
class ExAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 dropout_layer=None,
                 ex_module=EX_Module,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 draw_feat=False):
        super(ExAttention, self).__init__()
        self.dropout_layer = build_dropout(dropout_layer)
        self.ex_module = ex_module(in_channels=embed_dims,
                                   channels=embed_dims,
                                   norm_cfg=norm_cfg,
                                   draw_featmaps=draw_feat)

    def forward(self, x, hw_shape):
        x = nlc_to_nchw(x, hw_shape)
        ex_out = nchw_to_nlc(self.ex_module(x))

        return self.dropout_layer(ex_out)

class PSAAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 dropout_layer=None,
                 ex_module=None,
                 norm_cfg=dict(type='LN')):
        super(PSAAttention, self).__init__()
        self.dropout_layer = build_dropout(dropout_layer)
        self.psa_module = PSA_p(inplanes=embed_dims,
                                planes=embed_dims)

    def forward(self, x, hw_shape):
        x = nlc_to_nchw(x, hw_shape)
        ex_out = nchw_to_nlc(self.psa_module(x))

        return self.dropout_layer(ex_out)

class SEAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 dropout_layer=None,
                 ex_module=None,
                 norm_cfg=dict(type='LN')):
        super(SEAttention, self).__init__()
        self.dropout_layer = build_dropout(dropout_layer)
        self.se_layer = SELayer(channels=embed_dims)

    def forward(self, x, hw_shape):
        x = nlc_to_nchw(x, hw_shape)
        ex_out = nchw_to_nlc(self.se_layer(x))

        return self.dropout_layer(ex_out)

class ExTransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 index,
                 embed_dims,
                 token_mixer,
                 feedforward_channels,
                 draw_feat=False,
                 ex_module=EX_Module,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 with_cp=False):
        super(ExTransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.index = index
        self.draw_feat = draw_feat
        if index < 2:
            self.tokenmixer = InvertedResidual(
                in_channels=embed_dims,
                out_channels=embed_dims,
                stride=1,
                expand_ratio=2,
                act_cfg=act_cfg)
        else:
            self.tokenmixer = token_mixer(
                embed_dims=embed_dims,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                ex_module=ex_module,
                norm_cfg=norm_cfg,
                draw_feat=draw_feat)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):
        visualizer = Visualizer()
        def _inner_forward(x):
            if self.index < 2:
                x = nlc_to_nchw(self.norm1(x), hw_shape)
                x = self.tokenmixer(x)
                # draw_feat(x, visualizer, draw_feat)
                x = nchw_to_nlc(x)
            else:
                x = self.tokenmixer(self.norm1(x), hw_shape)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class ExMixVisionTransformer(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 token_mixers=ExAttention,
                 ex_module=EX_Module,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False,
                 draw_feat=False):
        super(ExMixVisionTransformer, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.with_cp = with_cp

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule
        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dims, 1))
        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                ExTransformerEncoderLayer(
                    index=i,
                    embed_dims=embed_dims_i,
                    # embed_dims=embed_dims_i + 1,
                    token_mixer=token_mixers,
                    draw_feat=draw_feat,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    ex_module=ex_module,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            # norm = build_norm_layer(norm_cfg, embed_dims_i + 1)[1]
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(ExMixVisionTransformer, self).init_weights()

    def forward(self, x, image):
        visualizer = Visualizer()
        if image is not None:
            ori_img = image.transpose(1, 2, 0)
            plt.imshow(ori_img)
            plt.show()

        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        outs = []
        for i, layer in enumerate(self.layers):
            # draw_feat(x, visualizer)
            x, hw_shape = layer[0](x)
            # if i == 0:
            #     cls_tokens = self.cls_token.expand(B, -1, -1)
            #     x = torch.cat((cls_tokens, x), dim=2)
            cls_token = x[..., 0]
            # draw_feat(nlc_to_nchw(x, hw_shape), visualizer, ori_img, draw_feat=True)
            # draw_feat(x, visualizer)

            for block in layer[1]:
                x = block(x, hw_shape)
                if i < 1:
                    draw_feat(nlc_to_nchw(x[..., 1:], hw_shape), visualizer, ori_img=None, draw_feat=True)
                    print("feature in layer{}".format(i))

            x = layer[2](x)
            # draw_feat(nlc_to_nchw(x, hw_shape), visualizer)
            x = nlc_to_nchw(x, hw_shape)
            # draw_feat(x, visualizer)
            if i in self.out_indices:
                # patch_token = x[..., 1:]
                # cls_token = x[..., 0]
                # outs.append([patch_token, cls_token])
                outs.append(x)
        #
        # return tuple(outs)
        return outs
