import cv2
import h5py
import glob
import torch
import random
import numpy as np
import os.path as osp
from PIL import Image

# from datasets import BaseLoader
from src.datasets import BaseLoader


class FDST(BaseLoader):
    def __init__(self, data_dir, transforms, crop_size=400, scale=8, mode='train'):
        super(FDST, self).__init__(data_dir, transforms, crop_size, scale, mode)

        self.time_step = scale

        self.get_file_path()

    def __getitem__(self, item):
        imgs, dens = self.load_files(item)

        for i in range(self.time_step):
            imgs[i] = self.transforms(imgs[i])
            dens[i] = torch.from_numpy(dens[i]).float().unsqueeze(0)

        imgs = torch.stack(imgs)
        dens = torch.stack(dens)

        if self.mode == 'train':
            # random horizontal flip
            if random.random() < 0.5:
                imgs = torch.flip(imgs, dims=[-1])
                dens = torch.flip(dens, dims=[-1])

            return imgs, dens
        elif self.mode == 'test':
            return imgs, dens

    def __len__(self):
        return len(self.img_path_list)

    def get_file_path(self):
        sub_data_dir = 'train_data' if self.mode == 'train' else 'test_data'
        img_dir = osp.join(self.data_dir, sub_data_dir, 'imgs')

        img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
        img_path_list = sorted(img_path_list, key=lambda s: int(osp.basename(s)[6:9]))
        img_path_list = sorted(img_path_list, key=lambda s: int(osp.basename(s)[1:4]))

        den_path_list = [p.replace('.jpg', '.h5').replace('imgs', 'dens') for p in img_path_list]

        for i in range(0, len(img_path_list), self.time_step):
            self.img_path_list.append(img_path_list[i: i + self.time_step])
            self.den_path_list.append(den_path_list[i: i + self.time_step])

    def load_files(self, item):
        imgs, dens = [], []
        img_path_list = self.img_path_list[item]
        den_path_list = self.den_path_list[item]

        for i in range(len(img_path_list)):
            img_path = img_path_list[i]
            den_path = den_path_list[i]

            img = Image.open(img_path).convert('RGB')

            try:
                with h5py.File(den_path, 'r') as hf:
                    den = np.asarray(hf['density'])
            except OSError:
                print('Failed to open file ', den_path)

            imgs.append(img)
            dens.append(den)

        return imgs, dens

