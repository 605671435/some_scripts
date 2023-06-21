# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
from mmengine.utils import scandir
import shutil
from mmengine.utils import ProgressBar

TRAINING_LEN = 634

def listdir(path):
    """ List files but remove hidden files from list """
    return [item for item in os.listdir(path) if item[0] != '.']

def main():
    files_path = 'data/lung_segmentation/'
    # out_dir 'data/lits17/'
    subfolders = ['train', 'val']
    train_dir = osp.join(files_path, 'images', 'training')
    val_dir = osp.join(files_path, 'images', 'validation')
    seg_dir = osp.join(files_path, 'masks')

    print('Generating images...')
    for filename in sorted(os.listdir(train_dir))[TRAINING_LEN:]:
        shutil.move(osp.join(train_dir, filename), osp.join(val_dir, filename))

    print('Generating annotations...')
    for filename in sorted(os.listdir(seg_dir))[:TRAINING_LEN]:
        img = mmcv.imread(osp.join(seg_dir, filename))
        # The annotation img should be divided by 128, because some of
        # the annotation imgs are not standard. We should set a
        # threshold to convert the nonstandard annotation imgs. The
        # value divided by 128 is equivalent to '1 if value >= 128
        # else 0'
        mmcv.imwrite(
            img[:, :, 0] // 128,
            osp.join(files_path, 'annotations', 'training',
                     osp.splitext(filename)[0] + '.png'))
    for filename in sorted(os.listdir(seg_dir))[TRAINING_LEN:]:
        img = mmcv.imread(osp.join(seg_dir, filename))
        mmcv.imwrite(
            img[:, :, 0] // 128,
            osp.join(files_path, 'annotations', 'validation',
                     osp.splitext(filename)[0] + '.png'))

    print('Done!')

def modify_filenames():
    files_path = 'data/lung_segmentation/'

    seg_dir = osp.join(files_path, 'annotations/training')
    for filename in sorted(os.listdir(seg_dir)):
        if filename.find('mask') != -1:
            prefix, suffix = filename.split('_mask')
            newname = prefix + suffix
            os.rename(osp.join(files_path, 'annotations/training', filename), osp.join(files_path, 'annotations/training', newname))

def resize_images():
    files_path = 'data/lung_segmentation/'
    subfolders = ['training', 'validation']

    size_list = []
    progressbar_subfolders = ProgressBar(len(subfolders))
    for folder in subfolders:
        img_dir = osp.join(files_path, 'images', folder)
        progressbar_imgdir = ProgressBar(len(os.listdir(img_dir)))
        for filename in os.listdir(img_dir):
            img = mmcv.imread(osp.join(img_dir, filename))
            img_size = img.shape
            if img_size not in size_list:
                size_list.append(img_size)
            progressbar_imgdir.update()
        progressbar_subfolders.update()
    print(size_list)
if __name__ == '__main__':
    # main()
    # modify_filenames()
    resize_images()