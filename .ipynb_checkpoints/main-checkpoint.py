import argparse
import torch
import time
import os
import shutil
from easydict import EasyDict as edict
from dataset.dataloader import get_dataloader
from lib.utils import load_obj, setup_seed
from lib.loss import CircleLoss, HardestContrastiveLoss
from model import load_model
from torch import optim
from trainer import Trainer
import pdb


setup_seed(0)

experiment_id = time.strftime('%m%d%H%M')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train/val/test')
    parser.add_argument('--max_epoch', type=int, default=100)

    parser.add_argument('--loss_type', type=str, default='CircleLoss/HardestContrastiveLoss')
#    parser.add_argument('--save_dir', type=str, default=f'./snapshot/{experiment_id}')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--root', type=str, default='/data/ghn/PredatorDataset/indoor')
    parser.add_argument('--train_info', type=str, default='./configs/train_info.pkl')
    parser.add_argument('--val_info', type=str, default='./configs/val_info.pkl')
    parser.add_argument('--test_info', type=str, default='./configs/3DMatch.pkl')
    parser.add_argument('--voxel_size', type=float, default=0.025)
    parser.add_argument('--search_radius', type=float, default=0.0375)
    parser.add_argument('--augment_noise', type=float, default=0.005)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--model', type=str, default='ResUNet_spconv')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_gamma', type=float, default=0.99)
    parser.add_argument('--verbose_freq', type=float, default=1000)

    parser.add_argument('--pos_margin', type=float, default=0.1)
    parser.add_argument('--neg_margin', type=float, default=1.4)
    parser.add_argument('--safe_radius', type=float, default=0.1)
    parser.add_argument('--pos_radius', type=float, default=0.0375)
    parser.add_argument('--max_points', type=float, default=256)

    parser.add_argument('--num_node', type=float, default=128)
    parser.add_argument('--neg_weight', type=float, default=1)
    parser.add_argument('--pos_thresh', type=float, default=0.1)
    parser.add_argument('--neg_thresh', type=float, default=1.4)
    parser.add_argument('--nn_max_n', type=int, default=500, help='The maximum number of features to find nearest neighbors in batch')


    args = parser.parse_args()
    dconfig = vars(args)
    config = edict(dconfig)
    config.save_dir = f'./snapshot/{config.loss_type}_{experiment_id}'
    os.makedirs(config.save_dir, exist_ok=True)
    shutil.copy2('main.py', config.save_dir)
    shutil.copy2('./dataset/ThreedMatch.py', config.save_dir)
    shutil.copy2('./model/resunet_spconv.py', os.path.join(config.save_dir, 'model.py'))
    shutil.copy2('run.sh', os.path.join(config.save_dir, 'run.sh'))


    if torch.cuda.device_count() > 0:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    ### Model Configuration ###
    model = load_model(config.model)
    config.model = model()
    print(config.model)

    ### Optimizer Configuration ###
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

        # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    ### Dataloader Configuration ###
    config.train_loader = get_dataloader(load_obj(config.train_info), config, data_augmentation=True)
    config.val_loader = get_dataloader(load_obj(config.val_info), config, data_augmentation=False)
    config.test_loader = get_dataloader(load_obj(config.test_info), config, data_augmentation=False)

    if config.loss_type == 'CircleLoss':
        config.desc_loss = CircleLoss(config)
    elif config.loss_type == 'HardestContrastiveLoss':
        config.desc_loss = HardestContrastiveLoss(config)
    else:
        raise NotImplementedError


    # pdb.set_trace()
    trainer = Trainer(config)
    if (config.phase == 'train'):
        trainer.train()
    elif (config.phase == 'val'):
        trainer.eval()
    elif (config.phase == 'test'):
        trainer.test()
    else:
        raise NotImplementedError

