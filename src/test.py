"""
Script for test
"""

import os.path as osp
import argparse
import torch.cuda

from misc.utilities import prepare, get_tester, get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training TransNet')

    parser.add_argument('--ckpt-path', type=str, help='checkpoint file path',
                        default='Experiments/RFSNet_FDST/ckpt_best.pth')
    parser.add_argument('--gpu-ids', type=int,
                        help='train config file path')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu_ids is not None:
            torch.cuda.set_device(args.gpu_ids)
        else:
            torch.cuda.set_device(0)

    base_dir = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
    args.ckpt_path = osp.join(base_dir, args.ckpt_path)

    return args


################################################################################
# main function
################################################################################
if __name__ == '__main__':
    ckpt_path = parse_args().ckpt_path

    # load checkpoint
    if torch.cuda.is_available():
        ckpt = torch.load(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # get configs
    cfg = ckpt['cfg']
    prev_base_dir = cfg.runner.base_dir
    curr_base_dir = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
    cfg.runner.base_dir = curr_base_dir
    cfg.runner.ckpt_dir = cfg.runner.ckpt_dir.replace(prev_base_dir, curr_base_dir)
    cfg.dataset.data_root = cfg.dataset.data_root.replace(prev_base_dir, curr_base_dir)

    # prepare log file
    prepare(cfg, mode='test')

    # load test set data loader
    val_loader = get_dataloader(cfg, mode='test')

    # Visualization option
    vis_options = {'img_raw':           False,
                   'gt_dot_map':        False,
                   'gt_density':        False,
                   'pr_density':        False,
                   'intermediate':      False}

    # test model
    tester = get_tester(ckpt, cfg, val_loader, vis_options)
    tester.run()
