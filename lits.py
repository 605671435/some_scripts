import os
from mmengine.utils import scandir
import os.path as osp
from mmseg.datasets.transforms import l  # noqa
def main():
    path = './data/LITS'
    split_names = ['training', 'validation']
    for dir in ['images', 'annotations']:
        for split in split_names:
            folder = osp.join(path, dir, split)
            for img_dir in scandir(folder, recursive=True):
                name = img_dir.split('-')[-1]
                os.rename(osp.join(folder, img_dir),
                          osp.join(folder, name))\


def main3():
    path = './data/LITS'
    split_names = ['training', 'validation']
    for dir in ['images', 'annotations']:
        for split in split_names:
            folder = osp.join(path, dir, split)
            for img_dir in scandir(folder, recursive=True):
                name = img_dir.split('-')[-1]
                os.rename(osp.join(folder, img_dir),
                          osp.join(folder, name))

def main2():
    path = './data/LITS'
    split_names = ['training', 'validation']
    for dir in ['images']:
        for split in split_names:
            folder = osp.join(path, dir, split)
            for img_dir in scandir(folder, recursive=True):
                results = dict(
                    img_path=osp.join(path, dir, split, img_dir),
                )
                transform1 = LoadBiomedicalImageFromFile()
                transform2 = LoadBiomedicalAnnotation()
                # transform2 = LoadAnnotations()
                results = transform1(copy.deepcopy(results))
                results = transform2(copy.deepcopy(results))
if __name__ == '__main__':
    main()