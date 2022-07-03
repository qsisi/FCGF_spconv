import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import square_distance
from sklearn.metrics import precision_recall_fscore_support
from scipy.spatial.distance import cdist
import pdb

class CircleLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """

    def __init__(self, configs, log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(CircleLoss, self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius
        self.pos_radius = configs.pos_radius  # just to take care of the numeric precision

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal)  # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * (~neg_mask).float()  # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight)  # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row) / self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col) / self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_recall(self, coords_dist, feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1) > 0).float().sum() + 1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist, dim=-1, index=sel_idx[:, None])[pos_mask.sum(-1) > 0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        # pdb.set_trace()
        recall = n_pred_pos / n_gt_pos
        return recall


    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """

        src_pcd = (torch.matmul(rot, src_pcd.transpose(0, 1)) + trans).transpose(0, 1)
        stats = dict()
#         pdb.set_trace()
        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondence[:, 0]] - tgt_pcd[correspondence[:, 1]], dim=1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        if (correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        src_idx = correspondence[:, 0]
        tgt_idx = correspondence[:, 1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        #######################
        # get L2 distance between source / target point cloud
        # pdb.set_trace()

        # coords_dist = cdist(src_pcd.detach().cpu(), tgt_pcd.detach().cpu())
        # coords_dist = torch.from_numpy(coords_dist).cuda()

        coords_dist = torch.sqrt(square_distance(src_pcd[None, :, :], tgt_pcd[None, :, :]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None, :, :], tgt_feats[None, :, :], normalised=True)).squeeze(0)
        print(feats_dist)
        # pdb.set_trace()
        # print(sum(torch.argmin(coords_dist, dim = -1) == torch.arange(256).cuda()))
        ##############################
        # get FMR and circle loss
        ##############################
        recall = self.get_recall(coords_dist, feats_dist)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)
        print(recall)
        stats['loss'] = circle_loss
        stats['recall'] = recall

        return stats

class HardestContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(HardestContrastiveLoss, self).__init__()
        self.num_node = config.num_node
        self.neg_weight = config.neg_weight
        self.pos_thresh = config.pos_thresh
        self.neg_thresh = config.neg_thresh
        self.pos_radius = config.search_radius

    def pdist(self, A, B, dist_type='L2'):
        if dist_type == 'L2':
            D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
            return torch.sqrt(D2 + 1e-7)
        elif dist_type == 'SquareL2':
            return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        else:
            raise NotImplementedError('Not implemented')

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans):
        stats = {}
        src_pcd = (torch.matmul(rot, src_pcd.T) + trans).T
#         pdb.set_trace()
        correspondence = correspondence.cpu()
        N0 = len(correspondence)
        if N0 > self.num_node:
            sel_ind = np.random.choice(N0, self.num_node, replace=False)
            sel_corr = correspondence[sel_ind, :]
        else:
            sel_corr = correspondence
        sel_src_pts = src_pcd[sel_corr[:, 0], :]
        sel_tgt_pts = tgt_pcd[sel_corr[:, 1], :]
        sel_src_feats = src_feats[sel_corr[:, 0], :]
        sel_tgt_feats = tgt_feats[sel_corr[:, 1], :]
        
        pts_dist = self.pdist(sel_src_pts, sel_tgt_pts)
        feats_dist = self.pdist(sel_src_feats, sel_tgt_feats)
           
        pos_mask = pts_dist < 0.0375
        neg_mask = pts_dist > 0.1
        furthest_positive, _ = torch.max(feats_dist * pos_mask.float(), dim = -1)
        closest_negative, _ = torch.min(feats_dist * neg_mask.float() + 1e5*((~neg_mask).float()), dim = -1)
#         pdb.set_trace()
#         print(feats_dist)
        recall = (furthest_positive < closest_negative).sum() / len(pos_mask) + 1e-7
        
        pos_loss = F.relu(furthest_positive - self.pos_thresh)
        neg_loss = F.relu(self.neg_thresh - closest_negative)
        loss = pos_loss.mean() + neg_loss.mean()
#         loss = F.relu(furthest_positive - self.pos_thresh) + F.relu(self.neg_thresh - closest_negative)
        
#         pdb.set_trace()
        stats['pos_loss'] = furthest_positive.mean()
        stats['neg_loss'] = closest_negative.mean()
        print(f'pos: {furthest_positive.mean()}  neg: {closest_negative.mean()} total: {loss.mean()} recall: {recall}')
        stats['recall'] = recall
        stats['loss'] = loss.mean()

        return stats








