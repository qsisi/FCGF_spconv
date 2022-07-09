"""
reference:
https://github.com/fabiopoiesi/dip/blob/master/benchmark_eth_pre.py
"""
import torch
import os
import numpy as np
import open3d as o3d
import argparse
from lib.utils import setup_seed
import spconv.pytorch as spconv
from demo import spconv_vox
from model.resunet_spconv import FCGF_spconv
from dataset.dataloader import collate_spconv_pair_fn

if __name__ == '__main__':
    setup_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ETH_root', default='/home/ghn/data/ETH', type=str)
    parser.add_argument('--voxel_size', default=0.06, type=float)
    parser.add_argument('--num_points', default=5000, type=int)
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--inlier_distance', default=0.10, type=float)
    args = parser.parse_args()

    names = ['gazebo_summer',
             'gazebo_winter',
             'wood_autumn',
             'wood_summer']
    FMR_RECALL = {name : [] for name in names}

    model = FCGF_spconv()
    checkpoint = torch.load('ckpt_30.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.cuda()

    for name in names:
        gt_file = open(os.path.join(args.ETH_root, name, 'PointCloud', 'gt.log'), 'r')
        gt = gt_file.readlines()
        nfrag = int(len(gt) / 5)

        for frag in range(nfrag):
            frag_ptr = frag * 5
            # read transformation
            T = np.empty((4, 4))
            T[0, :] = np.asarray(gt[frag_ptr + 1].split('\t'), dtype=np.float32)
            T[1, :] = np.asarray(gt[frag_ptr + 2].split('\t'), dtype=np.float32)
            T[2, :] = np.asarray(gt[frag_ptr + 3].split('\t'), dtype=np.float32)
            T[3, :] = np.asarray(gt[frag_ptr + 4].split('\t'), dtype=np.float32)

            info = gt[frag_ptr].split('\t')
            src_id = int(info[0])
            tgt_id = int(info[1])
            print(f'{name}\t{src_id}\t{tgt_id}')

            src_ply = o3d.io.read_point_cloud(os.path.join(args.ETH_root, name, 'PointCloud', f'cloud_bin_{src_id}.ply'))
            tgt_ply = o3d.io.read_point_cloud(os.path.join(args.ETH_root, name, 'PointCloud', f'cloud_bin_{tgt_id}.ply'))

            src_ply = src_ply.voxel_down_sample(args.voxel_size)
            tgt_ply = tgt_ply.voxel_down_sample(args.voxel_size)
            src_pcd = np.asarray(src_ply.points).astype(np.float32)
            tgt_pcd = np.asarray(tgt_ply.points).astype(np.float32)

            src_xyz, tgt_xyz, src_coords, tgt_coords, src_shape, tgt_shape = spconv_vox(src_pcd, tgt_pcd, args.voxel_size)

            src_features = torch.ones((len(src_coords), 1), dtype=torch.float32)
            tgt_features = torch.ones((len(tgt_coords), 1), dtype=torch.float32)
            list_data = [(src_xyz, tgt_xyz, src_coords, tgt_coords, src_features, tgt_features, torch.ones(1, 2),
                          np.eye(4), src_shape, tgt_shape, None, np.ones((6, 6)))]

            input_dict = collate_spconv_pair_fn(list_data)
            for k, v in input_dict.items():  # load inputs to device.
                if type(v) == list:
                    input_dict[k] = [item.cuda() for item in v]
                elif type(v) == torch.Tensor:
                    input_dict[k] = v.cuda(0)
                else:
                    pass

            src_sp_tensor = spconv.SparseConvTensor(input_dict['src_F'],
                                                    input_dict['src_C'].int(),
                                                    src_shape, batch_size=1)
            tgt_sp_tensor = spconv.SparseConvTensor(input_dict['tgt_F'],
                                                    input_dict['tgt_C'].int(),
                                                    tgt_shape, batch_size=1)
            ### get conv features ###
            with torch.no_grad():
                out_src = model(src_sp_tensor)
                out_tgt = model(tgt_sp_tensor)
            src_pcd = input_dict['pcd_src'].cpu().numpy()
            tgt_pcd = input_dict['pcd_tgt'].cpu().numpy()
            src_feats = out_src.features
            tgt_feats = out_tgt.features

            src_sel_idx = np.random.choice(len(src_pcd), min(len(src_pcd), args.num_points), replace=False)
            tgt_sel_idx = np.random.choice(len(tgt_pcd), min(len(tgt_pcd), args.num_points), replace=False)

            src_pcd = src_pcd[src_sel_idx]
            src_feats = src_feats[src_sel_idx]
            tgt_pcd = tgt_pcd[tgt_sel_idx]
            tgt_feats = tgt_feats[tgt_sel_idx]

            ### compute NN indices ###
            feats_M = torch.cdist(src_feats, tgt_feats, p=2)
            nn_mask = torch.logical_and(feats_M == feats_M.min(dim=-1, keepdim=True)[0],
                                        feats_M == feats_M.min(dim=-2, keepdim=True)[0])
            corr = nn_mask.nonzero().cpu().numpy()

            tgt_pcd_wrapped = (T[:3, :3] @ tgt_pcd.T + T[:3, 3:]).T
            distance = np.linalg.norm(src_pcd[corr[:, 0]] - tgt_pcd_wrapped[corr[:, 1]], axis=-1)

            ir = (distance < args.inlier_distance).sum() / len(distance)
            FMR_RECALL[name].append(ir > args.inlier_ratio_threshold)

    print("*"*50)
    recall_per_name = []
    for name in names:
        print(f'{name}\t{np.mean(FMR_RECALL[name])}')
        recall_per_name.append(np.mean(FMR_RECALL[name]))
    print(f'average recall -> {np.mean(recall_per_name)}')


