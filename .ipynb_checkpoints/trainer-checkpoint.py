import time, os, torch,copy
import numpy as np
import torch.nn as nn
import spconv.pytorch as spconv
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import validate_gradient
from tqdm import tqdm
import torch.nn.functional as F
import pdb



class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.start_epoch = 0
        self.max_epoch = config.max_epoch

        self.save_dir = config.save_dir
        self.tboard_dir = os.path.join(self.save_dir, 'tensorboard')
        self.model_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tboard_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = config.device
        self.verbose_freq = config.verbose_freq
        # self.max_points = args.max_points
        self.voxel_size = config.voxel_size

        self.model = config.model.to(self.device)

        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        # self.scheduler_freq = args.scheduler_freq
        # self.snapshot_freq = args.snapshot_freq

        # self.benchmark = args.benchmark
        # self.verbose_freq = args.verbose_freq

        self.desc_loss = config.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        if (config.pretrain != ''):
            self._load_pretrain(config.pretrain)

        self.train_loader = config.train_loader
        self.val_loader = config.val_loader
        self.test_loader = config.test_loader

    def train(self):
        print('start training...')
#         res = self.evaluate(self.start_epoch)
        for epoch in range(self.start_epoch, self.max_epoch):
            print(f"{'*'*30} Epoch {epoch} {'*'*30}")
            self.train_epoch(epoch + 1)  # train
            res = self.evaluate(epoch + 1) # val
            if res['loss'] < self.best_loss:
                self.best_loss = res['loss']
                self._snapshot(epoch + 1, 'best_loss')
            if res['recall'] > self.best_recall:
                self.best_acc = res['recall']
                self._snapshot(epoch + 1, 'best_recall')

            self.scheduler.step()

        # finish all epoch
        print("Training finish!... save training results")

    def train_epoch(self, epoch):

        stats_meter = self.stats_meter()
        badcase = 0
        train_loader_iter = self.train_loader.__iter__()
        num_iter = self.train_loader.__len__()
        self.model.train()
        for iter in tqdm(range(num_iter)):
            inputs = train_loader_iter.next()

            ### construct sparse tensor ###
            src_coords_min, src_coords_max = np.floor(np.percentile(inputs['src_C'], 0, axis=0)).astype(np.int32), \
                                             np.floor(np.percentile(inputs['src_C'], 100, axis=0)).astype(np.int32)
            tgt_coords_min, tgt_coords_max = np.floor(np.percentile(inputs['tgt_C'], 0, axis=0)).astype(np.int32), \
                                             np.floor(np.percentile(inputs['tgt_C'], 100, axis=0)).astype(np.int32)
            # pdb.set_trace()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)

            src_sp_tensor = spconv.SparseConvTensor(inputs['src_F'],
                                                    inputs['src_C'].int(),
                                                    (src_coords_max - src_coords_min)[1:], batch_size=1)
            tgt_sp_tensor = spconv.SparseConvTensor(inputs['tgt_F'],
                                                    inputs['tgt_C'].int(),
                                                    (tgt_coords_max - tgt_coords_min)[1:], batch_size=1)
            self.optimizer.zero_grad()
            try:
                # pdb.set_trace()
                out_src = self.model(src_sp_tensor)
                out_tgt = self.model(tgt_sp_tensor)
            except:
                badcase += 1
                continue
            # if iter % 200 == 0:
#             pdb.set_trace()
            ### get loss ###
            stats = self.desc_loss(inputs['pcd_src'], inputs['pcd_tgt'], out_src.features, out_tgt.features, inputs['correspondences'],
                                   inputs['rot'], inputs['trans'])
            # print(stats)
            # pdb.set_trace()
            c_loss = stats['loss']
            c_loss.backward()
            gradient_valid = validate_gradient(self.model)
            if gradient_valid:
                self.optimizer.step()
            else:
                print('gradient not valid\n')
            # update to stats_meter
            # pdb.set_trace()
            for key, value in stats.items():
                value = value.detach().cpu()
                stats_meter[key].update(value)
            del stats

            if iter == 0 or (iter + 1) % self.verbose_freq == 0:
                curr_iter = num_iter * (epoch - 1) + iter + 1
                self.writer.add_scalar(f'lr/lr', self._get_lr(), curr_iter)
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'train/{key}', value.avg, curr_iter)

        curr_iter = num_iter * (epoch - 1) + iter
        self.writer.add_scalar(f'lr/lr', self._get_lr(), curr_iter)
        for key, value in stats_meter.items():
            self.writer.add_scalar(f'train/{key}', value.avg, curr_iter)
            self.writer.add_scalar(f'train/invalid_pairs', badcase, curr_iter)


    def evaluate(self, epoch):
        stats_meter = self.stats_meter()
        self.model.eval()
        val_loader_iter = self.val_loader.__iter__()
        num_iter = self.val_loader.__len__()
        badcase = 0
        with torch.no_grad():
            for iter in tqdm(range(num_iter)):
                inputs = val_loader_iter.next()

                ### construct sparse tensor ###
                src_coords_min, src_coords_max = np.floor(np.percentile(inputs['src_C'], 0, axis=0)).astype(np.int32), \
                                                 np.floor(np.percentile(inputs['src_C'], 100, axis=0)).astype(np.int32)
                tgt_coords_min, tgt_coords_max = np.floor(np.percentile(inputs['tgt_C'], 0, axis=0)).astype(np.int32), \
                                                 np.floor(np.percentile(inputs['tgt_C'], 100, axis=0)).astype(np.int32)
                # pdb.set_trace()
                for k, v in inputs.items():  # load inputs to device.
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                src_sp_tensor = spconv.SparseConvTensor(inputs['src_F'],
                                                        inputs['src_C'].int(),
                                                        (src_coords_max - src_coords_min)[1:], batch_size=1)
                tgt_sp_tensor = spconv.SparseConvTensor(inputs['tgt_F'],
                                                        inputs['tgt_C'].int(),
                                                        (tgt_coords_max - tgt_coords_min)[1:], batch_size=1)
                try:
                    out_src = self.model(src_sp_tensor)
                    out_tgt = self.model(tgt_sp_tensor)
                except:
                    badcase += 1
                    continue
                # pdb.set_trace()
                ### get loss ###
                stats = self.desc_loss(inputs['pcd_src'], inputs['pcd_tgt'], out_src.features, out_tgt.features,
                                       inputs['correspondences'],
                                       inputs['rot'], inputs['trans'])
                # pdb.set_trace()
                # update to stats_meter
                for key, value in stats.items():
                    # value = value.detach().cpu()
                    stats_meter[key].update(value)
                # del stats
        self.writer.add_scalar(f'val/badcase', badcase, epoch - 1)
        for key, value in stats_meter.items():
            self.writer.add_scalar(f'val/{key}', value.avg, epoch - 1)
       # pdb.set_trace()
        res = {
                'loss': stats_meter['loss'].avg,
                'recall': stats_meter['recall'].avg
        }

        return res

    def eval(self):
        res = self.evaluate(0)
        for key, value in res.items():
            print(f'{key}: {value:2f}')


    def test(self):
        raise NotImplementedError


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
#	    pdb.set_trace()
            state = torch.load(resume)
            self.model.load_state_dict(state['state_dict'])
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']


            print(f'Successfully load pretrained model from {resume}!\n')
            print(f'Current best loss {self.best_loss}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        if name is None:
            filename = os.path.join(self.model_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.model_dir, f'model_{name}.pth')
        print(f"Save model to {filename}")
        torch.save(state, filename)

