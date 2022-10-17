# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
from mmengine.utils import scandir
import shutil
def listdir(path):
    """ List files but remove hidden files from list """
    return [item for item in os.listdir(path) if item[0] != '.']

def main():
    aptos_path = 'data/aptos/'
    subfolders = ['train', 'val', 'test']
    val_text = np.loadtxt(fname='data/aptos/meta/val.txt', dtype='str')
    val_text = val_text[:, 0]
    lenth = val_text.shape[0]
    train_files = 'data/aptos/train'

    moves = 0

    for i in range(lenth):
        filename = val_text[i]
        shutil.move(osp.join('data/aptos/train', filename), osp.join('data/aptos/val', filename))
        moves = moves + 1
    print(moves)
if __name__ == '__main__':
    main()
