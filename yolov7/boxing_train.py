import argparse
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils.general import (check_dataset, check_file, check_git_status, check_img_size,
    check_requirements, check_yaml, colorstr, get_latest_run, increment_path,
    print_mutation, set_logging, init_seeds)
from utils.torch_utils import select_device
from train import train

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='yolo_dataset/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    opt = parser.parse_args()
    return opt

def main(opt):
    # Set DDP variables
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.global_rank = int(os.environ.get('RANK', -1))

    # Set logging
    set_logging(opt.global_rank)

    # Resume
    if opt.resume and not opt.evolve:
        last = get_latest_run() if opt.resume == True else opt.resume
        if last and os.path.isfile(last):
            logging.info(f'Resuming training from {last}')
            opt.weights = last
        else:
            logging.info(f'No checkpoint found at {last}, starting from scratch')

    # Check Git status
    check_git_status()

    # Configure
    init_seeds(1 + opt.global_rank)
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)