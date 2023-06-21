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
        default='data/acdc',
        type=str,
        help='save path of the dataset.')
    args = parser.parse_args()
    return args


def main():
    args = pares_args()
    dataset_path = args.dataset_path
    save_path = args.save_path

    if not osp.exists(dataset_path):
        raise ValueError('The dataset path does not exist.'
                         ' Please enter a correct dataset path.')

    if not osp.exists(osp.join(dataset_path, 'training')) \
            or not osp.exists(osp.join(dataset_path, 'testing')):
        raise FileNotFoundError('The dataset structure is incorrect.'
                                ' Please check your dataset.')

    mkdir_or_exist(osp.join(save_path, 'img_dir'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir'))

    training_list = os.listdir(osp.join(dataset_path, 'training'))
    testing_list = os.listdir(osp.join(dataset_path, 'testing'))

    mkdir_or_exist(osp.join(save_path, 'img_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'img_dir/val'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/val'))

    progress_bar = ProgressBar(len(training_list))

    for patient in training_list:
        if 'patient' not in patient:
            continue

        images_list = os.listdir(osp.join(osp.join(dataset_path, 'training'), patient))

        images_list = [file for file in images_list if 'frame' in file]
        images_list = [file for file in images_list if 'gt' not in file]

        if len(images_list) == 0:
            continue

        for image in images_list:
            shutil.move(osp.join(osp.join(dataset_path, 'training'), patient, image),
                        osp.join(save_path, 'img_dir/train', image))
            shutil.move(osp.join(osp.join(dataset_path, 'training'), patient, image.replace('.nii.gz', '_gt.nii.gz')),
                        osp.join(save_path, 'ann_dir/train', image.replace('.nii.gz', '_gt.nii.gz')))
        progress_bar.update()

    progress_bar = ProgressBar(len(testing_list))

    for patient in testing_list:
        if 'patient' not in patient:
            continue

        images_list = os.listdir(osp.join(osp.join(dataset_path, 'testing'), patient))

        images_list = [file for file in images_list if 'frame' in file]
        images_list = [file for file in images_list if 'gt' not in file]

        if len(images_list) == 0:
            continue

        for image in images_list:
            shutil.move(osp.join(osp.join(dataset_path, 'testing'), patient, image),
                        osp.join(save_path, 'img_dir/val', image))
            shutil.move(osp.join(osp.join(dataset_path, 'testing'), patient, image.replace('.nii.gz', '_gt.nii.gz')),
                        osp.join(save_path, 'ann_dir/val', image.replace('.nii.gz', '_gt.nii.gz')))
        progress_bar.update()

    progress_bar = ProgressBar(len(testing_list))

if __name__ == '__main__':
    main()