import os
from mmengine.utils import scandir
import shutil
import os.path as osp
def main():
    ct_path = './data/LITS2/images/ct'
    label_path = './data/LITS2/annotations/label'
    imgs_path = './data/LITS2/images/'
    ann_path = './data/LITS2/annotations/'
    split_names = ['training', 'validation']
    list_dir = os.listdir(ct_path)
    list_dir.sort()
    samples_number = len(list_dir)
    training_number = int(samples_number * 0.8)
    for img_dir in list_dir[:training_number]:
        name = img_dir.split('-')[-1]
        ann_oldfile = img_dir.replace('volume', 'segmentation')
        shutil.move(osp.join(ct_path, img_dir), osp.join(imgs_path, split_names[0], name))
        shutil.move(osp.join(label_path, ann_oldfile), osp.join(ann_path, split_names[0], name))
    for img_dir in list_dir[training_number:]:
        name = img_dir.split('-')[-1]
        ann_oldfile = img_dir.replace('volume', 'segmentation')
        shutil.move(osp.join(ct_path, img_dir), osp.join(imgs_path, split_names[1], name))
        shutil.move(osp.join(label_path, ann_oldfile), osp.join(ann_path, split_names[1], name))
if __name__ == '__main__':
    main()