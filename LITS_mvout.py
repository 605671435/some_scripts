import os
from mmengine.utils import scandir
import shutil
import os.path as osp
import SimpleITK as sitk
import mmcv
import numpy as np
def main():
    # ct_path = './data/LITS-img/images/'
    label_path = './data/LITS2/annotations/'
    split_names = ['training', 'validation']
    for split in split_names:
        dir = osp.join(label_path, split)
        file_list = os.listdir(dir)
        for file in file_list:
            file_path = osp.join(dir, file)
            mask = sitk.ReadImage(file_path, sitk.sitkInt16)
            mask_array = sitk.GetArrayFromImage(mask)
            for i in range(mask_array.shape[0]):
                slice = mask_array[i]
                save_path = osp.join('./data/LITS-img/annotations/', split, file.split('.')[0] + '_' + str(i) + '.jpg')
                mmcv.imwrite(slice, save_path)

def main2():
    ct_path = './data/LITS_IMG_HD/images/'
    split_names = ['training', 'validation']
    for split in split_names:
        dir = osp.join(ct_path, split)
        file_list = os.listdir(dir)
        for file in file_list:
            file_path = osp.join(dir, file)
            dir_list = os.listdir(file_path)
            for img in dir_list:
                img_path = osp.join(file_path, img)
                shutil.move(img_path, dir)
            shutil.rmtree(file_path)

def main3():
    ct_path = './data/LITS_IMG_HD/batch2_img_hd/'
    dir_list = os.listdir(ct_path)
    for dir in dir_list:
        file_path = osp.join(ct_path, dir)
        file_list = os.listdir(file_path)
        for file in file_list:
            img_path = osp.join(file_path, file)
            new = file.split('-')[1:][0]
            os.rename(img_path, osp.join(file_path, new))
        new_dir = dir.split('-')[-1:][0]
        os.rename(file_path, osp.join(ct_path, new_dir))

def main4():
    label_path = './data/LITS_IMG_HD/annotations/'
    split_names = ['training', 'validation']
    for split in split_names:
        dir = osp.join(label_path, split)
        file_list = os.listdir(dir)
        for file in file_list:
            seg_path = osp.join(dir, file)
            seg_id = seg_path.split('-')[1].split('.')[0]
            seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array[seg_array > 0] = 1
            for i in range(seg_array.shape[0]):
                mmcv.imwrite(seg_array[i], osp.join(dir, f'{seg_id}_{str(i)}.jpg'))
            os.remove(seg_path)

def main4():
    ct_path = './data/LITS_IMG_HD/images/'
    label_path = './data/LITS_IMG_HD/annotations/'
    split_names = ['training', 'validation']
    for split in split_names:
        dir = osp.join(label_path, split)
        file_list = os.listdir(dir)
        for file in file_list:
            seg_path = osp.join(dir, file)
            seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            if np.max(seg_array) == 0:
                img_path = osp.join(ct_path, split, file)
                os.remove(seg_path)
                os.remove(img_path)
            elif np.max(seg_array) > 1:
                seg_array[seg_array > 0] = 1
                mmcv.imwrite(seg_array, seg_path)

if __name__ == '__main__':
    main4()