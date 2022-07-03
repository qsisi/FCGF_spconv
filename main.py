import argparse
import torch
import time
import os
import json
import shutil
from easydict import EasyDict as edict
from dataset.dataloader import get_dataloader
from lib.utils import setup_seed
from lib.loss import HardestContrastiveLoss
from model.resunet_spconv import FCGF_spconv
from torch import optim
from lib.trainer import Trainer
import pdb

experiment_id = time.strftime('%m%d%H%M')

if __name__ == '__main__':
    setup_seed(1234)
    parser = argparse.ArgumentParser()
    ### General Configuration
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--verbose_freq', type=float, default=100)
    ### Data Configuration
    parser.add_argument('--root', type=str, default='/home/ghn/data/FCGF_data/threedmatch')
    parser.add_argument('--voxel_size', type=float, default=0.025)
    parser.add_argument('--search_radius', type=float, default=0.0375)
    parser.add_argument('--rot_factor', type=int, default=4)
    parser.add_argument('--trans_scale', type=float, default=0.5)
    parser.add_argument('--jitter_noise', type=float, default=0.005)
    ### Optimizer Configuration
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_gamma', type=float, default=0.99)
    ### Loss Configuration
    parser.add_argument('--max_nodes', type=float, default=256)
    parser.add_argument('--pos_thresh', type=float, default=0.1)
    parser.add_argument('--neg_thresh', type=float, default=1.4)
    parser.add_argument('--pos_radius', type=float, default=0.0375)
    parser.add_argument('--safe_radius', type=float, default=0.1)
    args = parser.parse_args()
    dconfig = vars(args)
    config = edict(dconfig)

    config.snapshot_dir = f'./snapshot/{experiment_id}'
    config.ckpt_dir = os.path.join(config.snapshot_dir, 'checkpoints')
    config.tboard_dir = os.path.join(config.snapshot_dir, 'tensorboard')
    os.makedirs(config.snapshot_dir, exist_ok=False)
    os.makedirs(config.ckpt_dir, exist_ok=False)
    os.makedirs(config.tboard_dir, exist_ok=False)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    shutil.copy2('main.py', config.snapshot_dir)
    shutil.copy2('./dataset/ThreedMatch.py', config.snapshot_dir)
    shutil.copy2('./model/resunet_spconv.py', os.path.join(config.snapshot_dir, 'model.py'))

    ### Create Model ###
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.model = FCGF_spconv()
    # print(config.model)

    ### Create Optimizer ###
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )

    ### Create Scheduler ###
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    ### Create Dataloader ###
    config.train_loader = get_dataloader(config, split='train')
    config.val_loader = get_dataloader(config, split='val')

    ### Create Loss ###
    config.desc_loss = HardestContrastiveLoss(config)

    ### Create Trainer ###
    trainer = Trainer(config)

    if (config.phase == 'train'):
        trainer.train()
    elif (config.phase == 'val'):
        trainer.eval(epoch=-1)
    else:
        raise NotImplementedError

