import os
import torch
import numpy as np
import os.path as osp

from mmengine.visualization import Visualizer

from mit_ex import ExMixVisionTransformer
from vit import VisionTransformer
from mmseg.models.losses import CrossEntropyLoss

from mmengine.optim import build_optim_wrapper
from image_classifier import ImageClassifier
from mmcls.models.necks.gap import GlobalAveragePooling
from mmcls.models.heads.linear_head import LinearClsHead
from mmcls.models.heads.vision_transformer_head import VisionTransformerClsHead
from mmengine import FileClient
import mmcv
import pickle
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
def load_cifar():
    imgs = []
    gt_labels = []
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    file_client = FileClient.infer_client(uri='./data/cifar10/cifar-10-batches-py')
    # load the picked numpy arrays
    for file_name, _ in train_list:
        file_path = file_client.join_path('./data/cifar10/cifar-10-batches-py', file_name)
        content = file_client.get(file_path)
        entry = pickle.loads(content, encoding='latin1')
        imgs.append(entry['data'])
        if 'labels' in entry:
            gt_labels.extend(entry['labels'])
        else:
            gt_labels.extend(entry['fine_labels'])

    imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
    imgs = imgs.transpose((0, 2, 3, 1))  # convert to HWC

    data_list = []
    for img, gt_label in zip(imgs, gt_labels):
        info = {'img': img, 'gt_label': int(gt_label)}
        data_list.append(info)

    return data_list
def load_aptos():
    train_root = './data/aptos/train'
    train_text = np.loadtxt(fname='./data/aptos/meta/train.txt', dtype='str')
    train_list = train_text[:, 0]

    random_number = np.random.randint(low=0, high=len(train_list), size=1)
    image_name = train_list[random_number[0]]
    image = mmcv.imread(osp.join(train_root, image_name))
    h, w, _ = image.shape
    center = (int(h // 2), int(w // 2))
    h = min(h, w)
    left = center[0] - int(h // 2)
    up = center[1] - int(h // 2)
    image = image[left:left+h, up:up+h]
    image = mmcv.imresize(image, size=(224, 224)).transpose(2, 0, 1)
    label = int(train_text[random_number[0], 1])

    return image, label

def main():
    # data_list = load_cifar()
    # image, label = load_aptos()
    backbone = VisionTransformer(arch='deit-t',
                                img_size=224,
                                patch_size=16,
                                drop_rate=0.1,
                                init_cfg=[
                                    dict(
                                        type='Kaiming',
                                        layer='Conv2d',
                                        mode='fan_in',
                                        nonlinearity='linear')
                                ]).cuda(device='cuda:0')
    # neck = GlobalAveragePooling().cuda(device='cuda:0')
    # head = LinearClsHead(num_classes=10,
    #                          in_channels=512,
    #                          loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    #                          topk=1).cuda(device='cuda:0')
    head = VisionTransformerClsHead(num_classes=10,
                                    in_channels=192,
                                    loss=dict(
                                        type='LabelSmoothLoss', label_smooth_val=0.1,
                                        mode='classy_vision'),
                                    ).cuda(device='cuda:0')

    image_classifier = ImageClassifier(backbone, None, head)
    optim_wrapper = build_optim_wrapper(image_classifier,
                        cfg=dict(
                            type='OptimWrapper',
                            optimizer=dict(
                                type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)))
    losses = dict()
    iters = 30
    for iter in range(iters):
        image, label = load_aptos()
        # random_number = np.random.randint(low=0, high=len(data_list), size=1)
        # data = data_list[random_number[0]]
        # image = data['img'].transpose(2, 0, 1)
        # label = data['gt_label']
        inputs = torch.tensor(data=image, dtype=torch.float32).unsqueeze(0).cuda(device='cuda:0')
        label = torch.tensor(data=label, dtype=torch.int64).unsqueeze(0).cuda(device='cuda:0')

        pre_logits = image_classifier.forward(inputs, image)

        loss = dict()
        loss_decode = CrossEntropyLoss(loss_weight=1.0)
        loss['ce'] = loss_decode(
            pre_logits,
            label)
        losses.update(loss)
        optim_wrapper.update_params(loss['ce'])
        print("iter_{}:loss:{}".format(iter, loss))
if __name__ == '__main__':
    main()