import os
import sys
import numpy as np
import argparse
import open3d as o3d
import spconv.pytorch as spconv
from easydict import EasyDict as edict
from collections import defaultdict
from lib.timer import Timer, AverageMeter
from lib.benchmark_util import load_log, get_corr_from_dist_matrix, computeTransformationErr
from lib.utils import setup_seed, to_o3d_pcd, to_o3d_feats, to_array
from model.resunet_spconv import FCGF_spconv
from dataset.dataloader import get_dataloader
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_seed(0)
    parser.add_argument('--root', default=None, type=str, help='path to 3dmatch test set')
    parser.add_argument('--checkpoint',default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--voxel_size', default=0.025, type=float, help='voxel size to preprocess point cloud')
    parser.add_argument('--num_points', type=int, default=5000, help='Number of random keypoints for each scene')
    parser.add_argument('--mutual', default=False, action='store_true', help="whether to evaluation mutual inlier ratio")
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--inlier_distance', default=0.10, type=float)
    parser.add_argument('--distance_threshold', default=0.0375, type=float, help='ransac distance threshold')
    parser.add_argument('--ransac_n', default=3, type=int, help='ransac distance threshold')
    args = parser.parse_args()
    dconfig = vars(args)
    config = edict(dconfig)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    ### init model ###
    model = FCGF_spconv()
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    print(f'successfully loading ckpt in {config.checkpoint}')

    ### init dataset ###
    test_loader = get_dataloader(config, split='test')

    ir = defaultdict(list)
    fmr = defaultdict(list)
    rr = defaultdict(list)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    ### evaluate
    start.record()
    with torch.no_grad():
        data_iter = test_loader.__iter__()
        num_pairs = test_loader.__len__()
        for idx in range(num_pairs):
            input_dict = data_iter.next()
            scene, id1, id2 = input_dict['scene'].split('@')
            is_consecutive = True if abs(int(id1) - int(id2)) <= 1 else False  # exclude consecutive

            for k, v in input_dict.items():  # load inputs to device.
                if type(v) == list:
                    input_dict[k] = [item.to(device) for item in v]
                elif type(v) == torch.Tensor:
                    input_dict[k] = v.to(device)
                else:
                    pass

            ### construct sparse tensor ###
            src_shape = input_dict['grid_size'][:3]  # get sparse_shape
            tgt_shape = input_dict['grid_size'][3:]
            src_sp_tensor = spconv.SparseConvTensor(input_dict['src_F'],
                                                    input_dict['src_C'].int(),
                                                    src_shape, batch_size=1)
            tgt_sp_tensor = spconv.SparseConvTensor(input_dict['tgt_F'],
                                                    input_dict['tgt_C'].int(),
                                                    tgt_shape, batch_size=1)
            ### get conv features ###
            out_src = model(src_sp_tensor)
            out_tgt = model(tgt_sp_tensor)

            ### evaluation ###
            src_pcd = input_dict['pcd_src']
            tgt_pcd = input_dict['pcd_tgt']
            gt_trans = input_dict['tsfm']
            src_feats = out_src.features
            tgt_feats = out_tgt.features

            src_sel = np.random.choice(len(src_pcd), min(len(src_pcd), config.num_points), replace=False)
            tgt_sel = np.random.choice(len(tgt_pcd), min(len(tgt_pcd), config.num_points), replace=False)

            src_pcd = src_pcd[src_sel]
            tgt_pcd = tgt_pcd[tgt_sel]
            src_feats = src_feats[src_sel]
            tgt_feats = tgt_feats[tgt_sel]
            feats_dist = torch.cdist(src_feats, tgt_feats, p=2)
            corr = get_corr_from_dist_matrix(feats_dist, config.mutual)
            # ### get inlier ratio ###
            src_pcd_sel = src_pcd[corr[:, 0]]
            tgt_pcd_sel = tgt_pcd[corr[:, 1]]
            src_pcd_wrapped = (gt_trans[:3, :3] @ src_pcd_sel.transpose(1, 0) + gt_trans[:3, 3][:, None]).transpose(1, 0)
            distance = torch.norm(src_pcd_wrapped - tgt_pcd_sel, p=2, dim=-1)
            num_inliers = torch.sum(distance < config.inlier_distance)
            inlier_ratio = num_inliers / len(distance)
            feature_match_recall = inlier_ratio > config.inlier_ratio_threshold
            ir[scene].append(inlier_ratio.cpu().item())
            fmr[scene].append(feature_match_recall.cpu().item())

            ### ransac registration ###
            if not is_consecutive:   # only do ransac for those non-consecutive pairs
                result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
                    to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), to_o3d_feats(src_feats), to_o3d_feats(tgt_feats),
                    config.distance_threshold,
                    o3d.registration.TransformationEstimationPointToPoint(False), config.ransac_n,
                    [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                     o3d.registration.CorrespondenceCheckerBasedOnDistance(config.distance_threshold)],
                    o3d.registration.RANSACConvergenceCriteria(50000, 1000))
                pred_trans = result_ransac.transformation
                info = input_dict['info']
                p = computeTransformationErr(np.linalg.inv(to_array(gt_trans)) @ pred_trans, to_array(info))
                rr[scene].append(p <= 0.2 ** 2)
                print(f'{scene}\t{id1}\t{id2}\t{idx + 1}/{num_pairs}\tir:{inlier_ratio:.4f}\tfmr:{feature_match_recall}\trr:{p <= 0.2 ** 2}')
            else:
                print(f'{scene}\t{id1}\t{id2}\t{idx + 1}/{num_pairs}\tir:{inlier_ratio:.4f}\tfmr:{feature_match_recall}')

    end.record()
    t = start.elapsed_time(end)
    print('********evaluation results********')
    print(f'Benchmark: 3DMatch\tSample: {config.num_points}\tMutual: {config.mutual}\t tot_time: {t/1000/60:.2f}m')
    tot_ir, tot_fmr, tot_rr = [], [], []
    for scene in test_loader.dataset.scene_list:
        scene_ir = np.mean(ir[scene])
        scene_fmr = np.mean(fmr[scene])
        scene_rr = np.mean(rr[scene])
        print(f'{scene:50}\tir: {scene_ir:5.4f}\tfmr: {scene_fmr:5.4f}\trr: {scene_rr:5.4f}')
        tot_ir.append(scene_ir)
        tot_fmr.append(scene_fmr)
        tot_rr.append(scene_rr)
    print(f'Average Inlier Ratio: {np.mean(tot_ir):.4f}\tFeature Match Recall: {np.mean(tot_fmr):.4f}'
          f'\tRegistration Recall: {np.mean(tot_rr):.4f}')




