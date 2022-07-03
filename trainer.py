import time
import os
import torch
import numpy as np
import torch.nn as nn
import spconv.pytorch as spconv
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from tqdm import tqdm
import torch.nn.functional as F
import pdb

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.start_epoch = 0
        self.max_epoch = config.max_epoch
        self.snapshot_dir = config.snapshot_dir
        self.tboard_dir = config.tboard_dir
        self.ckpt_dir = config.ckpt_dir
        self.device = config.device
        self.verbose_freq = config.verbose_freq
        self.voxel_size = config.voxel_size
        self.model = config.model.to(self.device)
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.desc_loss = config.desc_loss
        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=config.tboard_dir)
        if (config.pretrain != ''):
            self._load_pretrain(config.pretrain)

        self.train_loader = config.train_loader
        self.val_loader = config.val_loader

    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            print(f"{'*'*30} Epoch {epoch} {'*'*30}")
            self.train_epoch(epoch)  # train
            self.scheduler.step()
            self._snapshot()

            res = self.eval(epoch)    # val
            if res['loss'] < self.best_loss:
                self.best_loss = res['loss']
                self._snapshot(name='best_loss')
            if res['recall'] > self.best_recall:
                self.best_acc = res['recall']
                self._snapshot(name='best_recall')

        # finish all epoch
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        stats_meter = self.stats_meter()
        invalid_inputs = 0

        train_loader_iter = self.train_loader.__iter__()
        num_iter = self.train_loader.__len__()
        self.model.train()
        pbar = tqdm(range(num_iter))
        for iter in pbar:
            inputs = train_loader_iter.next()

            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)

            ### construct sparse tensor ###
            src_shape = inputs['grid_size'][:3]  # get sparse_shape
            tgt_shape = inputs['grid_size'][3:]
            src_sp_tensor = spconv.SparseConvTensor(inputs['src_F'],
                                                    inputs['src_C'].int(),
                                                    src_shape, batch_size=1)
            tgt_sp_tensor = spconv.SparseConvTensor(inputs['tgt_F'],
                                                    inputs['tgt_C'].int(),
                                                    tgt_shape, batch_size=1)
            try:   # This is due to some inputs will get zero coordinates in several dimensions during down-sampling
                out_src = self.model(src_sp_tensor)
                out_tgt = self.model(tgt_sp_tensor)
            except:
                invalid_inputs += 1
                continue

            ### get loss ###
            stats = self.desc_loss(inputs['pcd_src'], inputs['pcd_tgt'], out_src.features, out_tgt.features, inputs['correspondences'],
                                   inputs['tsfm'])

            c_loss = stats['loss']
            c_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            ### update to stats_meter ###
            for key, value in stats.items():
                value = value.detach().cpu()
                stats_meter[key].update(value)

            if iter == 0 or (iter + 1) % self.verbose_freq == 0:
                curr_iter = num_iter * epoch + iter + 1
                self.writer.add_scalar(f'lr/lr', self._get_lr(), curr_iter)
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'train/{key}', value.avg, curr_iter)

            pbar.set_description(f"loss:{stats_meter['loss'].avg.item():.4f} "
                                 f"recall:{stats_meter['recall'].avg.item():.4f} "
                                 f"pos:{stats_meter['pos_loss'].avg.item():.4f} "
                                 f"neg:{stats_meter['neg_loss'].avg.item():.4f} ")

        self.writer.add_scalar(f'train/invalid_pairs', invalid_inputs, epoch)

    def eval(self, epoch):
        eval_meter = self.stats_meter()
        self.model.eval()
        val_loader_iter = self.val_loader.__iter__()
        num_iter = self.val_loader.__len__()
        invalid_inputs = 0
        with torch.no_grad():
            for iter in tqdm(range(num_iter)):
                inputs = val_loader_iter.next()

                for k, v in inputs.items():  # load inputs to device.
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                ### construct sparse tensor ###
                src_shape = inputs['grid_size'][:3]
                tgt_shape = inputs['grid_size'][3:]
                src_sp_tensor = spconv.SparseConvTensor(inputs['src_F'],
                                                        inputs['src_C'].int(),
                                                        src_shape, batch_size=1)
                tgt_sp_tensor = spconv.SparseConvTensor(inputs['tgt_F'],
                                                        inputs['tgt_C'].int(),
                                                        tgt_shape, batch_size=1)
                try:
                    out_src = self.model(src_sp_tensor)
                    out_tgt = self.model(tgt_sp_tensor)
                except:
                    invalid_inputs += 1
                    continue

                ### get loss ###
                stats = self.desc_loss(inputs['pcd_src'], inputs['pcd_tgt'], out_src.features, out_tgt.features,
                                       inputs['correspondences'],
                                       inputs['tsfm'])

                ### update to stats_meter ###
                for key, value in stats.items():
                    value = value.detach().cpu()
                    eval_meter[key].update(value)

        res = {
            'loss': eval_meter['loss'].avg,
            'recall': eval_meter['recall'].avg
        }

        if epoch == -1:
            print(f"val -> epoch {epoch} loss: {res['loss']} recall: {res['recall']}")

        else:
            self.writer.add_scalar(f'val/invalid_pairs', invalid_inputs, epoch)
            for key, value in eval_meter.items():
                self.writer.add_scalar(f'val/{key}', value.avg, epoch)
            return res

    def stats_dict(self):
        stats = dict()
        stats['pos_loss'] = 0.
        stats['neg_loss'] = 0.
        stats['loss'] = 0.
        stats['recall'] = 0.  # feature match recall, divided by number of ground truth pairs
        return stats

    def stats_meter(self):
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume)
            self.model.load_state_dict(state['state_dict'])
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']

            print(f'Successfully load pretrained model from {resume}!')
            print(f'Current best loss {self.best_loss}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _snapshot(self, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        if name is None:
            filename = os.path.join(self.ckpt_dir, f'checkpoint.pth')
        else:
            filename = os.path.join(self.ckpt_dir, f'model_{name}.pth')
        print(f"Save model to {filename}")
        torch.save(state, filename)

