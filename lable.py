import mmcv
from mmengine.utils import scandir
import os.path as osp
def main():
    path = './data/lung_segmentation/images'
    out_dir = osp.join(path, 'meta')

    split_names = ['training', 'validation']
    # class_names = ['Covid', 'Normal', 'Viral Pneumonia']
    for split in split_names:
        folder = osp.join(path, split)
        filenames = []
        classnames = []
        for img_dir in scandir(folder, recursive=True):
            lable = img_dir.split(sep='_')[-1].split('.')[0]
            filenames.append(img_dir + ' ' + lable)
            # classnames.append(classname)
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)

if __name__ == '__main__':
    main()