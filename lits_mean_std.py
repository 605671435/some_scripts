import os

import mmcv
import numpy
import numpy as np
from mmengine.utils import scandir
import os.path as osp
import SimpleITK as sitk
from mmengine.utils.progressbar import ProgressBar, track_parallel_progress

def main():
    path = './data/LITS2'
    split_names = ['training', 'validation']
    first = True
    for dir in ['images']:
        for split in split_names:
            folder = osp.join(path, dir, split)
            for img_dir in scandir(folder, recursive=True):
                ct = sitk.ReadImage(osp.join(folder, img_dir), sitk.sitkInt16)
                ct_array = sitk.GetArrayFromImage(ct)
                if first is True:
                    pixel = ct_array
                    first = False
                else:
                    pixel = numpy.append(ct_array, pixel, axis=0)
    pixel.mean()

def aptos_mean():
    path = './data/aptos/train'
    # first = True
    file_list = os.listdir(path)
    n = len(file_list)
    progress_bar = ProgressBar(n)
    [r_mean, g_mean, b_mean] = [0, 0, 0]
    for file in file_list:
        image = mmcv.imread(osp.join(path, file))
        # [r, g, b] = image[..., :]
        r_mean += numpy.mean(image[..., 0])
        g_mean += numpy.mean(image[..., 1])
        b_mean += numpy.mean(image[..., 2])
        progress_bar.update()
    r_mean /= n
    g_mean /= n
    b_mean /= n
    mean = [r_mean, g_mean, b_mean]
    print(mean)

def aptos_std():
    path = './data/aptos/train'
    # first = True
    file_list = os.listdir(path)
    n = len(file_list)
    mean = [19, 56, 106]
    [r_std, g_std, b_std] = [0, 0, 0]
    progress_bar = ProgressBar(int(n*0.25))
    for file in file_list[:int(n*0.25)]:
        image = mmcv.imread(osp.join(path, file))
        # [r, g, b] = image[..., :]
        hw_shape = image.shape[:2]
        r = image[..., 0]
        r = r - np.ones(shape=hw_shape) * mean[0]
        r = r ** 2
        r_std += np.sum(r)

        g = image[..., 1]
        g = g - np.ones(shape=hw_shape) * mean[1]
        g = g ** 2
        g_std += np.sum(g)

        b = image[..., 2]
        b = b - np.ones(shape=hw_shape) * mean[2]
        b = b ** 2
        b_std += np.sum(b)
        progress_bar.update()
    r_std = (r_std // n) ** 0.5
    g_std = (g_std // n) ** 0.5
    b_std = (b_std // n) ** 0.5
    std = [r_std, g_std, b_std]
    print(std)

def aptos_std2():
    path = './data/aptos/train'
    first = True
    file_list = os.listdir(path)
    n = len(file_list)
    mean = [19, 56, 106]
    [r_std, g_std, b_std] = [0, 0, 0]
    progress_bar = ProgressBar(int(n*0.25))
    for file in file_list[:int(n*0.25)]:
        image = mmcv.imread(osp.join(path, file))
        if first is True:
            pixel_r = image[..., 0]
            pixel_g = image[1]
            pixel_b = image[2]
            first = False
        else:
            pixel_r = numpy.append(image[0], pixel_r, axis=0)
            pixel_g = numpy.append(image[1], pixel_g, axis=0)
            pixel_b = numpy.append(image[2], pixel_b, axis=0)
        progress_bar.update()
    numpy.std(pixel_r)
    numpy.std(pixel_g)
    numpy.std(pixel_b)
    print(pixel_r, pixel_g, pixel_b)

if __name__ == '__main__':
    aptos_std2()