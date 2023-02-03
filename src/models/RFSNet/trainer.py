# import libraries
import logging
import numpy as np
from os.path import isfile
from importlib import import_module
from mmcv.utils import Config
from typing import List

# import torch modules
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data.dataloader import DataLoader

# import local modules
from src.models import BaseTrainer
from src.misc.utilities import MultiAverageMeters


class Trainer(BaseTrainer):
    def __init__(self,
                 cfg: Config,
                 data_loaders: List[DataLoader]):

        super(Trainer, self).__init__(cfg, data_loaders)

        if "RFSNet_" in cfg.model.name:
            RFSNet = import_module(f"src.models.RFSNet.{cfg.model.name}").RFSNet
        else:
            from src.models.RFSNet import RFSNet

        model = RFSNet(cfg).float()
        optimizer = Adam(model.parameters(), lr=cfg.optimizer.lr,
                         weight_decay=cfg.optimizer.weight_decay)

        self.model = model.to(self.device)
        self.model_name = cfg.model.name
        self.model_type = cfg.model.type if cfg.model.type else 'V'
        self.optimizer = optimizer

        # resume from checkpoint
        if isfile(cfg.runner.resume):
            self.load_from_resume()

        # loss function(s)
        self.loss_functions = dict()
        self.loss_functions["MSE"] = MSELoss(reduction="sum").to(self.device)

        # other stuff
        self.meter = MultiAverageMeters(num=2)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logging.info("\tTrainable parameters: %.3fM" % parameters)

    def epoch_train_step(self, i, data):
        imgs, dens = data
        imgs = imgs.float().to(self.device)
        dens = dens.float().to(self.device)

        outs = self.model(imgs)

        # loss
        loss = torch.tensor(0.0).to(self.device)
        mse_loss = self.loss_functions["MSE"](outs["density"], dens)
        loss += mse_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.meter.update([loss.item(), mse_loss.item()])

        if (i + 1) % self.cfg.runner.print_freq == 0:
            logging.info("\tit: {:4d} | sum_loss: {:10.4f} | mse_loss: {:10.4f}".format(
                i + 1, self.meter[0].avg, self.meter[1].avg))

        self.meter.reset()

    def epoch_val_step(self, data):
        imgs, dens = data
        imgs = imgs.float().to(self.device)

        outs = self.model(imgs)     # [1, time_step, 1, H, W]

        gt = dens[0].sum(-1).sum(-1).sum(-1)
        gt = gt.cpu().detach().numpy()
        pr = outs["density"][0].sum(-1).sum(-1).sum(-1)
        pr = pr.cpu().detach().numpy()

        return gt, pr
