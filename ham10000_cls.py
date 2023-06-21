# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
from mmengine.utils import mkdir_or_exist, ProgressBar
import shutil

def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert HAM10000 dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset-path', type=str, help='HAM10000 dataset path.')
    parser.add_argument(
        '--save-path',
        default='data/ham10000',
        type=str,
        help='save path of the dataset.')
    args = parser.parse_args()
    return args


def main():
    args = pares_args()
    dataset_path = args.dataset_path
    save_path = args.save_path

    dx = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    if not osp.exists(dataset_path):
        raise ValueError('The dataset path does not exist.'
                         ' Please enter a correct dataset path.')

    if not osp.exists(osp.join(dataset_path, 'images')) \
            or not osp.exists(osp.join(dataset_path, 'labels')):
        raise FileNotFoundError('The dataset structure is incorrect.'
                                ' Please check your dataset.')

    mkdir_or_exist(osp.join(save_path, 'img_dir'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir'))

    images_list = os.listdir(osp.join(dataset_path, 'images'))
    samples_len = len(images_list)
    train_len = int(samples_len * 0.85)

    mkdir_or_exist(osp.join(save_path, 'img_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'img_dir/val'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/val'))

    progress_bar = ProgressBar(samples_len)

    for image in images_list[0:train_len]:
        shutil.move(osp.join(dataset_path, 'images', image),
                    osp.join(save_path, 'img_dir/train', image))
        shutil.move(osp.join(dataset_path, 'labels', image.replace('.jpg', '_segmentation.png')),
                    osp.join(save_path, 'ann_dir/train', image.replace('.jpg', '_segmentation.png')))
        progress_bar.update()

    for image in images_list[train_len:]:
        shutil.move(osp.join(dataset_path, 'images', image),
                    osp.join(save_path, 'img_dir/val', image))
        shutil.move(osp.join(dataset_path, 'labels', image.replace('.jpg', '_segmentation.png')),
                    osp.join(save_path, 'ann_dir/val', image.replace('.jpg', '_segmentation.png')))
        progress_bar.update()

def fix_label():
    args = pares_args()
    save_path = args.save_path

    train_dir = osp.join(save_path, 'ann_dir/train')
    val_dir = osp.join(save_path, 'ann_dir/val')

    for filename in sorted(os.listdir(train_dir)):
        img = mmcv.imread(osp.join(train_dir, filename), flag='grayscale')
        mmcv.imwrite(
            img,
            osp.join(train_dir, filename))
    for filename in sorted(os.listdir(val_dir)):
        img = mmcv.imread(osp.join(val_dir, filename), flag='grayscale')
        mmcv.imwrite(
            img,
            osp.join(val_dir, filename))

if __name__ == '__main__':
    # main()
    fix_label()