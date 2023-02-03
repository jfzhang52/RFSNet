"""
Script for training
"""

import argparse
import os.path as osp
import torch.autograd
from mmcv.utils import Config

from misc.utilities import prepare, get_dataloader, get_trainer, setup_seed

################################################################################
# configuration
################################################################################
# set random seed for reproducibility
manualSeed = 1
# manualSeed = random.randint(1, 10000) # use if you want new results
setup_seed(manualSeed)

torch.autograd.set_detect_anomaly(True)
# torch.hub.set_dir('~/.cache/torch/hub')     # torch case dir


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training Counting Model')

    parser.add_argument('--config', type=str,
                        default='./configs/RFSNet/v1/UCSD.py', help='config file path for training')
    parser.add_argument('--gpu-ids', type=int,
                        help='gpu ids for training')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu_ids is not None:    # TODO: Implementing distributed training
            torch.cuda.set_device(args.gpu_ids)
        else:
            torch.cuda.set_device(0)

    if args.config != '':
        config_file = args.config
        args = Config.fromfile(config_file)
        args.config_file = {'path': config_file}

    if args.runner.base_dir == '':
        # args.runner.base_dir = osp.dirname(osp.abspath(__file__))
        args.runner.base_dir = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir))

    return args


################################################################################
# main function
################################################################################
if __name__ == '__main__':
    # get configurations and initialization
    cfg = parse_args()
    prepare(cfg, mode='train')

    # load dataset
    data_loaders = get_dataloader(cfg)

    # train model
    trainer = get_trainer(cfg, data_loaders)
    trainer.run()
