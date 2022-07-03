import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class HardestContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(HardestContrastiveLoss, self).__init__()
        self.max_nodes = config.max_nodes
        self.pos_thresh = config.pos_thresh # 0.1
        self.neg_thresh = config.neg_thresh # 1.4
        self.pos_radius = config.search_radius # 0.0375
        self.safe_radius = config.safe_radius # 0.1

    def pdist(self, A, B, dist_type='L2'):
        if dist_type == 'L2':
            D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
            return torch.sqrt(D2 + 1e-7)
        elif dist_type == 'SquareL2':
            return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        else:
            raise NotImplementedError('Not implemented')

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, tsfm):
        stats = {}
        src_pcd = (torch.matmul(tsfm[:3, :3], src_pcd.T) + tsfm[:3, 3][:, None]).T

        c_dist = torch.norm(src_pcd[correspondence[:, 0]] - tgt_pcd[correspondence[:, 1]], dim=-1)
        c_select = c_dist < self.pos_radius
        correspondence = correspondence[c_select]
        correspondence = correspondence.cpu()
        N0 = len(correspondence)
        if N0 > self.max_nodes:
            sel_ind = np.random.choice(N0, self.max_nodes, replace=False)
            sel_corr = correspondence[sel_ind, :]
        else:
            sel_corr = correspondence
        sel_src_pts = src_pcd[sel_corr[:, 0], :]
        sel_tgt_pts = tgt_pcd[sel_corr[:, 1], :]
        sel_src_feats = src_feats[sel_corr[:, 0], :]
        sel_tgt_feats = tgt_feats[sel_corr[:, 1], :]

        pts_dist = self.pdist(sel_src_pts, sel_tgt_pts)
        feats_dist = self.pdist(sel_src_feats, sel_tgt_feats)
           
        pos_mask = pts_dist < self.pos_radius
        neg_mask = pts_dist > self.safe_radius
        a, _ = torch.max(feats_dist * pos_mask.float(), dim=-1)
        b, _ = torch.min(feats_dist + 1e5 * (~neg_mask), dim=-1)
        c, _ = torch.min(feats_dist + 1e5 * (~neg_mask), dim=0)

        diff = a - (b + c) / 2
        accuracy = (diff < 0).sum() / diff.shape[0]
        average_negative = (torch.sum(feats_dist, dim=-1) - a) / (diff.shape[0] - 1)
        pos_a = F.relu(a - self.pos_thresh).pow(2).mean()
        neg_b = F.relu(self.neg_thresh - b).pow(2).mean()
        neg_c = F.relu(self.neg_thresh - c).pow(2).mean()
        loss = pos_a + (neg_b + neg_c) / 2

        stats['pos_loss'] = a.mean()
        stats['neg_loss'] = average_negative.mean()
        stats['recall'] = accuracy
        stats['loss'] = loss.mean()

        return stats








