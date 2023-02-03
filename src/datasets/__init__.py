import os
import cv2
import h5py
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional, Compose
from typing import Union, Tuple, Any

from src.misc.utilities import resize_dot_map


class BaseLoader(Dataset):
    def __init__(self,
                 data_dir: str,
                 transforms: Compose,
                 crop_size: Union[int, Tuple[int, int]] = 400,
                 scale: int = 8,
                 mode: str = "train") -> None:

        super(BaseLoader, self).__init__()

        self.data_dir = data_dir
        self.transforms = transforms
        self.crop_size = crop_size
        self.scale = scale
        self.mode = mode

        self.img_path_list = []
        self.dot_path_list = []
        self.den_path_list = []

        self.img = None
        self.den = None
        self.dot = None

    def __getitem__(self, item: int) -> Tuple[Any, ...]:
        self.img, self.dot, self.den = self.load_files(item)

        if self.mode == 'train':
            return self.train_transform()
        elif self.mode == 'test':
            return self.test_transform()

    def __len__(self) -> int:
        return len(self.img_path_list)

    def get_file_path(self) -> None:
        pass

    def load_files(self, item: int) -> Tuple[Any, ...]:
        img_path = self.img_path_list[item]
        dot_path = self.dot_path_list[item]
        den_path = self.den_path_list[item]

        img = Image.open(img_path).convert('RGB')

        with h5py.File(dot_path, 'r') as hf:
            dot = np.asarray(hf['dot'])

        if os.path.isfile(den_path):
            with h5py.File(den_path, 'r') as hf:
                den = np.asarray(hf['density'])
        else:
            den = dot

        return img, dot, den

    def test_transform(self) -> Tuple[Any, ...]:
        # image
        self.img = self.transforms(self.img)

        # dot map
        self.dot = self.dot[np.newaxis, :, :]

        # density map
        h = self.den.shape[1] // self.scale
        w = self.den.shape[0] // self.scale
        self.den = cv2.resize(self.den, (h, w), interpolation=cv2.INTER_LINEAR)
        self.den = self.den * (self.scale ** 2)
        self.den = self.den[np.newaxis, :, :]

        return self.img, self.dot, self.den

    def train_transform(self) -> Tuple[Any, ...]:
        self.random_resize()
        self.random_crop()
        self.random_flip()
        self.random_gamma()
        self.random_gray()

        self.img = self.transforms(self.img)

        self.den = cv2.resize(
            self.den, (self.crop_size // self.scale, self.crop_size // self.scale),
            interpolation=cv2.INTER_LINEAR)
        self.den *= self.scale**2
        self.den = self.den[np.newaxis, :, :]
        self.den = torch.from_numpy(self.den).float()

        points = np.array(np.where(self.dot > 0)).transpose()
        permutation = [1, 0]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        points[:] = points[:, idx]
        points = torch.from_numpy(points).float()

        self.dot = resize_dot_map(self.dot, 1 / self.scale, 1 / self.scale)
        self.dot = self.dot[np.newaxis, :, :]
        self.dot = torch.from_numpy(self.dot).float()

        return self.img, self.dot, self.den, points

    def random_resize(self) -> None:
        w, h = self.img.size
        short = min(w, h)

        if short < 512:
            scale = 512 / short
            w = round(w * scale)
            h = round(h * scale)

            self.img = self.img.resize((w, h), Image.BILINEAR)

            self.den = cv2.resize(self.den, (w, h), interpolation=cv2.INTER_LINEAR)
            self.den = self.den / (scale * scale)

            gt_count = np.sum(self.dot)
            self.dot = resize_dot_map(self.dot, scale, scale)
            assert int(np.sum(self.dot)) == int(gt_count)

        scale_x = random.uniform(0.8, 1.2)
        scale_y = random.uniform(0.8, 1.2)
        w = round(w * scale_y)
        h = round(h * scale_x)

        self.img = self.img.resize((w, h), Image.BILINEAR)

        self.den = cv2.resize(self.den, (w, h), interpolation=cv2.INTER_LINEAR)
        self.den = self.den / (scale_x * scale_y)

        gt_count = np.sum(self.dot)
        self.dot = resize_dot_map(self.dot, scale_x, scale_y)
        assert int(np.sum(self.dot)) == int(gt_count)

    def random_crop(self) -> None:
        w, h = self.img.size
        ch, cw = self.crop_size, self.crop_size
        dh = random.randint(0, h - ch)
        dw = random.randint(0, w - cw)

        self.img = self.img.crop((dw, dh, dw + cw, dh + ch))
        self.den = self.den[dh:dh + ch, dw:dw + cw]
        self.dot = self.dot[dh:dh + ch, dw:dw + cw]

    def random_flip(self) -> None:
        if random.random() < 0.5:
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
            self.dot = self.dot[:, ::-1]  # equivalent to np.fliplr
            self.den = self.den[:, ::-1]

    def random_gamma(self) -> None:
        if random.random() < 0.3:
            gamma = random.uniform(0.5, 1.5)
            self.img = functional.adjust_gamma(self.img, gamma)

    def random_gray(self) -> None:
        if random.random() < 0.1:
            self.img = functional.to_grayscale(self.img, num_output_channels=3)
