# -*- coding: utf-8 -*-
"""
utilities
"""

# import libraries
import os
import time
import h5py
import random
import shutil
import logging
import os.path as osp
import matplotlib.pyplot as plt
from importlib import import_module
from typing import Union, List

# import scientific libraries
import numpy as np
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

# import torch modules
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

# import mmcv modules
from mmcv.utils import Config


################################################################################
# change the learning rate according to epoch.
################################################################################
def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int) -> None:
    if (epoch + 1) % 100 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5


################################################################################
# set the random seed.
################################################################################
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


################################################################################
# dataloader collate function
################################################################################
def train_collate(batch):
    transposed_batch = list(zip(*batch))

    if len(transposed_batch) == 4:
        img = torch.stack(transposed_batch[0], 0)
        dot = torch.stack(transposed_batch[1], 0)
        den = torch.stack(transposed_batch[2], 0)
        pts = transposed_batch[3]

        return img, dot, den, pts

    elif len(transposed_batch) == 2:
        img = torch.stack(transposed_batch[0], 0)
        cnt = torch.stack(transposed_batch[1], 0)

        return img, cnt

    elif len(transposed_batch) == 8:
        img_1 = torch.stack(transposed_batch[0], 0)
        dot_1 = torch.stack(transposed_batch[1], 0)
        den_1 = torch.stack(transposed_batch[2], 0)
        pts_1 = transposed_batch[3]

        img_2 = torch.stack(transposed_batch[4], 0)
        dot_2 = torch.stack(transposed_batch[5], 0)
        den_2 = torch.stack(transposed_batch[6], 0)
        pts_2 = transposed_batch[7]

        return img_1, dot_1, den_1, pts_1, \
               img_2, dot_2, den_2, pts_2


################################################################################
# prepare data loader
################################################################################
def get_dataloader(cfg: Config,
                   mode: str = "train") -> Union[DataLoader, List[DataLoader]]:
    scale = cfg.dataset.scale
    crop_size = cfg.dataset.crop_size
    mean = cfg.dataset.img_norm_cfg.mean
    std = cfg.dataset.img_norm_cfg.std

    data_root = osp.join(cfg.runner.base_dir, cfg.dataset.data_root)
    cfg.dataset.data_root = data_root

    batch_size = cfg.dataset.batch_size
    num_workers = cfg.runner.num_workers

    data_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    dataset = cfg.dataset.name.upper()
    CCDataset = import_module(f"src.datasets.{dataset}")
    CCDataset = getattr(CCDataset, dataset)

    val_dataset = CCDataset(data_root, data_transform,
                            scale=scale, mode='test')
    val_loader = DataLoader(val_dataset, batch_size=1,
                            num_workers=num_workers, pin_memory=True)

    if mode == 'train':
        train_dataset = CCDataset(data_root, data_transform, scale=scale,
                                  crop_size=crop_size, mode='train')
        train_loader = DataLoader(train_dataset, collate_fn=train_collate,
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

        return [train_loader, val_loader]
    else:   # mode == 'test'
        return val_loader


################################################################################
# prepare the model trainer
################################################################################
def get_trainer(cfg, data_loaders):
    network = cfg.model.name.split('_')[0]
    Trainer = import_module(f"src.models.{network}.trainer").Trainer
    trainer = Trainer(cfg, data_loaders)

    return trainer


################################################################################
# prepare the model trainer
################################################################################
def get_tester(ckpt, cfg, val_loader, vis_options):
    network = cfg.model.name.split('_')[0]
    Tester = import_module(f"src.models.{network}.tester").Tester
    tester = Tester(ckpt, cfg, val_loader, vis_options)

    return tester


################################################################################
# get logger and copy current environment
################################################################################
def prepare(cfg: Config, mode: str = "train") -> None:
    if mode == 'train':
        if not osp.isfile(cfg.runner.resume):
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

            if cfg.runner.ckpt_dir == '':
                cfg.runner.ckpt_dir = osp.join(
                    cfg.runner.base_dir, 'Experiments',
                    cfg.model.name + '_' + cfg.dataset.name.upper() + '_' + current_time)
                os.makedirs(cfg.runner.ckpt_dir, exist_ok=True)

            copy_cur_env(cfg.runner.base_dir, cfg.runner.ckpt_dir + '/code')
        else:
            if cfg.runner.ckpt_dir == '':
                cfg.runner.ckpt_dir = osp.abspath(osp.dirname(cfg.runner.resume))

    log_file_name = 'train.log' if mode == 'train' else 'test.log'
    log_file_path = osp.join(cfg.runner.ckpt_dir, log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_formatter = logging.Formatter(
        fmt='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    # if mode == 'train':
    #     file_handler = logging.FileHandler(log_file_path, mode='a')
    # else:
    #     file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logger_formatter)
    logger.addHandler(console_handler)

    if mode == 'train':
        if not osp.isfile(cfg.runner.resume):
            logging.info('{} {}'.format('> TRAINING CONFIG ', '-' * 60))
            logging.info('\n{}'.format(cfg.pretty_text))
            logging.info('{} {}'.format('> START TRAINING  ', '-' * 60))
    else:
        logging.info('{} {}'.format('> TEST CONFIG  ', '-' * 60))
        logging.info('\n{}'.format(cfg.pretty_text))
        logging.info('{} {}'.format('> START TESTING   ', '-' * 60))


################################################################################
# resize the tensor
################################################################################
def resize(input, size, flag=False):
    scale_factor_height = input.shape[-2] / size[-2]
    scale_factor_width = input.shape[-1] / size[-1]
    input = F.interpolate(input=input, size=size,
                          mode='bilinear', align_corners=True)
    if flag:
        input *= scale_factor_height * scale_factor_width
    return input


################################################################################
# save code
################################################################################
def copy_cur_env(work_dir, dst_dir, exception=None):
    if exception is None:
        exception = [".git", ".idea", ".backup", ".ipynb_checkpoints", ".DS_Store",
                     "__pycache__", "data", "logs", "Experiments"]

    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)
    else:
        raise IOError("Dir \'" + dst_dir + "\' already exists!")

    for filename in os.listdir(work_dir):
        file = osp.join(work_dir, filename)
        dst_file = osp.join(dst_dir, filename)

        if filename not in exception:
            # print(f"[{filename}] | [{file}] -> [{dst_file}]")
            if osp.isdir(file):
                shutil.copytree(file, dst_file)
            elif osp.isfile(file):
                shutil.copyfile(file, dst_file)


################################################################################
# plot the density map.
################################################################################
def plot_density(dm, dm_dir, img_name, if_count=False):
    # type: (np.ndarray, str, str, bool) -> None
    assert osp.isdir(dm_dir)

    dm = dm[0, 0, :, :]
    count = np.sum(dm)

    dm = dm / np.max(dm + 1e-20)

    dm_frame = plt.gca()
    plt.imshow(dm, 'jet')

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    if if_count:
        dm_file_name = img_name.split('.')[0] + '_cnt_%.2f.jpg' % count
    else:
        dm_file_name = img_name.split('.')[0]
    plt.savefig(osp.join(dm_dir, dm_file_name),
                bbox_inches='tight', pad_inches=0, dpi=150)

    plt.close()


################################################################################
# save density map as .h5 file
################################################################################
def save_h5(gt_warped, save_dir, img_name, name='dot'):
    # type: (np.ndarray, str, str, str) -> None
    gt_warped = gt_warped[0, 0, :, :]

    save_name = img_name.replace('.jpg', '.h5')
    save_path = osp.join(save_dir, save_name)

    with h5py.File(save_path, 'w') as hf:
        hf[name] = gt_warped


################################################################################
# draw grid on image
################################################################################
def draw_grid(x, grid_size=0, grid_interval=15, grid_color=1):
    # type: (torch.Tensor, int, int, int) -> torch.Tensor
    b, c, h, w = x.shape

    grid_color = torch.tensor([grid_color])

    if grid_size:
        dx = int(w / grid_size)
        dy = int(h / grid_size)

        x[:, :, ::dy, :] = grid_color
        x[:, :, :, ::dx] = grid_color
    if grid_interval:
        x[:, :, ::grid_interval, :] = grid_color
        x[:, :, :, ::grid_interval] = grid_color
    return x


################################################################################
# generate density map with adaptive kernel size
################################################################################
def generate_density(dot_map):
    shape = dot_map.shape
    density = np.zeros(shape, dtype=np.float32)

    gt_count = np.count_nonzero(dot_map)

    if gt_count == 0:
        return density

    points = np.array(np.where(dot_map > 0), dtype=np.float64).transpose()

    tree = scipy.spatial.KDTree(points.copy(), leafsize=2048)
    distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        point2d = np.zeros(shape, dtype=np.float32)
        point2d[int(pt[0]), int(pt[1])] = 1.

        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(shape)) / 2. / 2.

        density += gaussian_filter(point2d, sigma, mode='constant')

    return density


################################################################################
# Resize the dot map with the given scale
################################################################################
def resize_dot_map(dot_map, scale_x, scale_y):
    h, w = dot_map.shape
    h, w = round(h * scale_x), round(w * scale_y)

    points_old = np.array(np.where(dot_map > 0), dtype=np.float64)
    points_new = np.zeros_like(points_old)
    points_new[0] = points_old[0] * scale_x
    points_new[1] = points_old[1] * scale_y
    points_new, points_old = points_new.transpose(), points_old.transpose()

    res = np.zeros([h, w], dtype=np.float64)

    for k in range(len(points_new)):
        i, j = np.floor(points_old[k] + 0.5)
        x, y = np.floor(points_new[k] + 0.5)

        x, y = min(x, h - 1), min(y, w - 1)

        res[int(x), int(y)] += dot_map[int(i), int(j)]

    return res


################################################################################
# average meter
################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.cur_val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = float(self.sum / self.count)


class MultiAverageMeters(list):
    def __init__(self, num):
        super(MultiAverageMeters, self).__init__()
        self.num = num

        for i in range(num):
            self.append(AverageMeter())

    def reset(self):
        for i in range(self.num):
            self.__getitem__(i).reset()

    def update(self, values):
        for i in range(self.num):
            self.__getitem__(i).update(values[i])


################################################################################
# Timer
################################################################################
class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls


################################################################################
# Reverse image transformation
################################################################################
class de_normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


################################################################################
# Visualize density map and raw image simultaneously
################################################################################
def vis_density(im, density, save_path, show_img=True, alpha=0.6):
    dm_frame = plt.gca()
    if show_img:
        plt.imshow(im)
        plt.imshow(density, 'jet', alpha=alpha)
    else:
        plt.imshow(density, 'jet')

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()


################################################################################
# Visualize dot map and raw image simultaneously
################################################################################
def vis_dot_map(im, dot_map, save_path, show_img=True):
    pts = np.asarray(np.where(dot_map > 0)).transpose()
    permutation = [1, 0]
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    pts[:] = pts[:, idx]

    dm_frame = plt.gca()
    if show_img:
        plt.imshow(im)

    dm_frame.axes.get_yaxis().set_visible(False)
    dm_frame.axes.get_xaxis().set_visible(False)
    dm_frame.spines['top'].set_visible(False)
    dm_frame.spines['bottom'].set_visible(False)
    dm_frame.spines['left'].set_visible(False)
    dm_frame.spines['right'].set_visible(False)

    plt.scatter(pts[:, 0], pts[:, 1], s=5, c='r')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()
