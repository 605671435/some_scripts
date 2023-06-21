import os

import mmcv
import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from ex_attention import EX_Module
from mmengine.visualization import Visualizer
from mmseg.registry import MODELS
from mit_ex import ExMixVisionTransformer
from mmseg.models.losses import CrossEntropyLoss
from mmseg.models.utils import resize
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim import build_optim_wrapper
from encoder_decoder import Encoder_Decoder
import PIL

def draw_feat(feature, visualizer, ori_img=None):
    drawn_img = visualizer.draw_featmap(feature.squeeze(0),
                                        overlaid_image=ori_img,
                                        channel_reduction='select_max',
                                        resize_shape=(256, 256))
    visualizer.show(drawn_img, wait_time=1)
def arr_to_img(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    new = (arr - min) * (1. / (max - min)) * 255
    new = new.astype(np.uint8)
    # img = Image.fromarray(new)
    return new
def main():
    visualizer = Visualizer()

    backbone = ExMixVisionTransformer(in_channels=3,
                                        embed_dims=64,
                                        num_stages=4,
                                        num_layers=[3, 4, 6, 3],
                                        num_heads=[1, 2, 5, 8],
                                        patch_sizes=[7, 3, 3, 3],
                                        out_indices=(0, 1, 2, 3),
                                        mlp_ratio=4,
                                        drop_path_rate=0.1).cuda(device='cuda:0')
    decode_head = MODELS.build(cfg=dict(
                                    type='SegformerHead',
                                    in_channels=[64, 128, 320, 512],
                                    in_index=[0, 1, 2, 3],
                                    channels=256,
                                    dropout_ratio=0.1,
                                    num_classes=2,
                                    norm_cfg=None,
                                    align_corners=False,
                                    loss_decode=dict(
                                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))).cuda(device='cuda:0')
    encoder_decoder = Encoder_Decoder(backbone, decode_head)
    optim_wrapper = build_optim_wrapper(encoder_decoder,
                        cfg=dict(
                            type='OptimWrapper',
                            optimizer=dict(
                                type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.01)))
    losses = dict()
    iters = 30
    for iter in range(iters):
        images_path = './data/lung_segmentation/images/training/'
        images_list = os.listdir(images_path)
        random_number = np.random.randint(low=0, high=600, size=1)
        image_path = images_list[random_number[0]]
        label_path = './data/lung_segmentation/annotations/training/'

        image = mmcv.imread(img_or_path=osp.join(images_path, image_path)).transpose([2, 0, 1])

        r = mmcv.imresize(image[0], size=(256, 256), backend='pillow')
        g = mmcv.imresize(image[1], size=(256, 256), backend='pillow')
        b = mmcv.imresize(image[2], size=(256, 256), backend='pillow')
        image = np.array([r, g, b])

        label = mmcv.imread(img_or_path=osp.join(label_path, image_path), flag='grayscale')
        label = mmcv.imresize(label, size=(256, 256), backend='pillow')

        inputs = torch.tensor(data=image, dtype=torch.float32).unsqueeze(0).cuda(device='cuda:0')
        seg_label = torch.tensor(data=label, dtype=torch.long).cuda(device='cuda:0')

        seg_logits = encoder_decoder.forward(inputs)
        # x = backbone.forward(inputs)
        # seg_logits = decode_head.forward(x)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape,
            mode='bilinear',
            align_corners=False)
        loss = dict()
        loss_decode = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
        loss['ce'] = loss_decode(
            seg_logits,
            seg_label.unsqueeze(0),
            weight=None,
            ignore_index=255)
        losses.update(loss)
        optim_wrapper.update_params(loss['ce'])
        print("iter_{}:loss:{}".format(iter, loss))
if __name__ == '__main__':
    main()