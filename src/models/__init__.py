import os
import time
import logging
import numpy as np
import os.path as osp
import PIL.Image
from typing import List, Dict, Any, Union
from mmcv.utils import Config

import torch
import torch.nn as nn
from torchvision import transforms

from src.datasets import BaseLoader
from src.misc.utilities import Timer, adjust_learning_rate, de_normalize, plot_density


class BaseTrainer(nn.Module):
    def __init__(
            self,
            cfg: Config,
            data_loaders: List[BaseLoader]
    ) -> None:
        super(BaseTrainer, self).__init__()

        self.cfg = cfg
        self.train_loader, self.val_loader = data_loaders

        if torch.cuda.is_available():
            # gpus = str(cfg.runner.device)
            # os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.epoch = 0
        self.start_epoch = 0

        self.BEST_MAE = 300.0
        self.BEST_MSE = 300.0
        self.CURR_MAE = self.BEST_MAE
        self.CURR_MSE = self.BEST_MSE

        self.timer = {'train': Timer(), 'val': Timer()}

    def run(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.runner.max_epochs):
            # ----- training -----
            self.epoch = epoch
            adjust_learning_rate(self.optimizer, self.epoch)

            logging.info("> Epoch: {:4d}/{:4d} {:s}".format(
                self.epoch + 1, self.cfg.runner.max_epochs, '-' * 60))

            self.timer["train"].tic()
            self.epoch_train()
            self.timer["train"].toc()

            self.save_model(self.epoch, 'ckpt.pth')

            logging.info("  - [Train] cost time: {:5.1f}s | lr: {:.4e}".format(
                self.timer["train"].diff, self.optimizer.param_groups[0]["lr"]))

            # ----- validation -----
            if (epoch + 1) % self.cfg.runner.val_freq == 0:
                self.timer["val"].tic()
                self.epoch_val()
                self.timer['val'].toc()

                if self.CURR_MAE < self.BEST_MAE:
                    self.BEST_MAE, self.BEST_MSE = self.CURR_MAE, self.CURR_MSE
                    self.save_model(epoch, "ckpt_best.pth")

                logging.info("  - [Val]   cost time: {:5.1f}s | MAE: {:6.2f}, MSE: {:6.2f} "
                             "(BEST: {:6.2f}/{:6.2f})".format(
                                self.timer["val"].diff, self.CURR_MAE, self.CURR_MSE,
                                self.BEST_MAE, self.BEST_MSE))

    def epoch_train(self) -> None:
        self.model.train()

        for (i, data) in enumerate(self.train_loader):
            self.epoch_train_step(i, data)

    def epoch_train_step(
            self,
            i: int,
            data: BaseLoader
    ) -> None:
        pass

    def epoch_val(self) -> None:
        self.model.eval()

        mae_sum, mse_sum = 0.0, 0.0
        N = 0

        with torch.no_grad():
            for data in self.val_loader:
                gt, pr = self.epoch_val_step(data)

                N += gt.shape[0]
                mae_sum += np.sum(np.abs(gt - pr))
                mse_sum += np.sum(np.abs(gt - pr) ** 2)

        self.CURR_MAE = mae_sum / N
        self.CURR_MSE = np.sqrt(mse_sum / N)

    def epoch_val_step(
            self,
            data: BaseLoader
    ) -> List[torch.Tensor]:
        pass

    def load_from_resume(self) -> None:
        logging.info('\t==> Resuming from checkpoint: {:s}'.format(
            self.cfg.runner.resume))

        state = torch.load(self.cfg.runner.resume)
        self.model.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optim'])

        self.start_epoch = state['epoch']
        self.BEST_MAE = state['mae']
        self.BEST_MSE = state['mse']

    def save_model(
            self,
            epoch: int,
            name: str
    ) -> None:
        state = {'net': self.model.state_dict(),
                 'optim': self.optimizer.state_dict(),
                 'epoch': epoch,
                 'mae': self.BEST_MAE,
                 'mse': self.BEST_MSE,
                 'cfg': self.cfg}
        torch.save(state, os.path.join(self.cfg.runner.ckpt_dir, name))

    def img_transform(
            self,
            img: PIL.Image
    ) -> torch.Tensor:
        mean = self.cfg.dataset.img_norm_cfg.mean
        std = self.cfg.dataset.img_norm_cfg.std
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

        return t(img)

    def img_restoration(
            self,
            img: PIL.Image
    ) -> torch.Tensor:
        mean = self.cfg.dataset.img_norm_cfg.mean
        std = self.cfg.dataset.img_norm_cfg.std
        t = transforms.Compose([de_normalize(mean, std),
                                transforms.ToPILImage()])

        return t(img)


class BaseTester(nn.Module):
    def __init__(
            self,
            ckpt: Any,
            cfg: Config,
            val_loader: BaseLoader,
            vis_options: Dict
    ) -> None:
        super(BaseTester, self).__init__()

        self.ckpt = ckpt
        self.cfg: Config = cfg
        self.val_loader: BaseLoader = val_loader
        self.vis_options: Dict = vis_options

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.MAE = 0.
        self.MSE = 0.
        self.timer = Timer
        self.restore_transform = transforms.Compose([
            de_normalize(mean=self.cfg.dataset.img_norm_cfg.mean,
                         std=self.cfg.dataset.img_norm_cfg.std),
            transforms.ToPILImage()])

        self.vis_dirs = self.create_dir()

    def run(self) -> None:
        logging.info("\t{:>4s} | {:<10s} | {:>4s} | {:>12s} | {:>8s}".format(
            "no.", "name", "gt", "prediction", "diff"))

        count = 0
        count_0 = 0
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            for data in self.val_loader:
                # img_path_list = self.val_loader.dataset.img_path_list[count_0]
                img_path_list = self.val_loader.dataset.img_path_list[count_0]
                count_0 += 1
                imgs, dens = data
                T = len(img_path_list)

                imgs = imgs.float().to(self.device)
                dens = dens.float().to(self.device)

                outs = self.model(imgs)
                outs = outs["density"]

                for i in range(T):
                    img_name = osp.basename(img_path_list[i])
                    count += 1

                    img = imgs[0][i].unsqueeze(0)
                    den = dens[0][i].unsqueeze(0)
                    out = outs[0][i].unsqueeze(0)

                    gt = torch.sum(den).cpu().detach().numpy()
                    pr = torch.sum(out).cpu().detach().numpy()

                    self.MAE += np.abs(gt - pr)
                    self.MSE += (gt - pr) ** 2

                    self.visualization(img_name, img, den, out)

                    logging.info("\t{:>4d} | {:<10s} | {:>4d} | {:>12.2f} | {:>8.2f}".format(
                        count, img_name.split('.')[0], int(gt + 0.5), pr, pr - gt))

        self.MAE = self.MAE / count
        self.MSE = np.sqrt(self.MSE / count)
        end_time = time.time()
        total_time = end_time - start_time

        logging.info('{} {}'.format('> END TESTING     ', '-' * 60))
        logging.info('\t[Test Result] MAE: {:6.2f}\tMSE: {:6.2f}'.format(self.MAE, self.MSE))
        logging.info("\t{:d} images in {:8.2f} seconds\tFPS: {:8.4f}".format(
            count, total_time, total_time / count))

    def create_dir(self) -> Dict:
        ckpt_dir = self.cfg.runner.ckpt_dir
        infer_dir = osp.join(ckpt_dir, 'inference')
        # os.makedirs(infer_dir, exist_ok=True)

        vis_dirs = {}

        for k, v in self.vis_options.items():
            if v:
                dirname = osp.join(infer_dir, k)
                os.makedirs(dirname, exist_ok=True)
                vis_dirs[k] = dirname

        return vis_dirs

    def visualization(
            self,
            img_name: str,
            img: torch.Tensor,
            gt: Union[List[torch.Tensor], torch.Tensor],
            pr_density: torch.Tensor
    ) -> None:
        if isinstance(gt, List):
            gt_dot_map, gt_density = gt
        else:
            gt_dot_map, gt_density = None, gt

        # save img
        if self.vis_options['img_raw']:
            img_raw = self.restore_transform(img.cpu().squeeze())
            img_raw.save(osp.join(self.vis_dirs['img_raw'], img_name))

        # save dot map
        if self.vis_options['gt_dot_map']:
            gt_dot_map = gt_dot_map.cpu().detach().numpy()
            plot_density(gt_dot_map, self.vis_dirs['gt_dot_map'], img_name)

        # save density map
        if self.vis_options['gt_density']:
            gt_density = gt_density.cpu().detach().numpy()
            plot_density(gt_density, self.vis_dirs['gt_density'], img_name)

        # save prediction
        if self.vis_options['pr_density']:
            pr_density = pr_density.cpu().detach().numpy()
            plot_density(pr_density, self.vis_dirs['pr_density'], img_name)
