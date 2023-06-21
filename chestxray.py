import os
from mmengine.utils import scandir
import os.path as osp
import shutil
import numpy as np
from mmengine.utils.progressbar import ProgressBar
def mv_rm():
    path = './data/small-xray/val'
    list = os.listdir(path)
    for i in range(len(list)):
        dir = osp.join(path, list[i])
        for img_dir in scandir(dir, recursive=True):
            shutil.move(osp.join(dir, img_dir), path)
        shutil.rmtree(dir)

def find(f, new_f, path):
    now_line = f.readline()
    now_name = now_line.split('.png')[0]
    for img_dir in scandir(path, recursive=True):
        img_name = str(img_dir.split('.png')[0])
        if now_name == img_name:
            new_f.writelines(now_line)

def train_label():
    # train_number = 11212
    train_number = 75312
    path = './data/chestxray14/val'
    label = './data/chestxray14/Xray14_val_official.txt'
    new_label = './data/chestxray14/val.txt'

    f = open(label, 'r+')
    new_f = open(new_label, 'r+')
    progress_bar = ProgressBar(train_number)
    now_list = []
    for img_dir in scandir(path, recursive=True):
        img_name = str(img_dir.split('.png')[0])
        now_list.append(img_name)
    now_list.sort()
    for i in range(train_number):
        now_line = f.readline()
        now_name = now_line.split('.png')[0]

        # track_parallel_progress(find, (f, new_f, path), 16)
        if len(now_list) > 0:
            for i in range(len(now_list)):
                img_name = now_list[i]
                now_prefix = int(now_name.split('_')[0])
                img_prefix = int(img_name.split('_')[0])
                if img_prefix > now_prefix:
                    break
                else:
                    if now_name == img_name:
                        now_list.remove(now_list[i])
                        new_f.writelines(now_line)
                        break
            progress_bar.update()
        else:
            break
    new_f.close()
    f.close()

def split_files():
    path = './data/val'
    files_list = os.listdir(path)
    files_list.sort()
    all_number = len(files_list)
    split_number = int(all_number * 0.25)
    progress_bar = ProgressBar(split_number)
    for file in files_list[:split_number]:
        shutil.copy(osp.join(path, file), '/home/s316/Workspace/dataset/chestxray14/val')
        progress_bar.update()
if __name__ == '__main__':
    train_label()