import open3d as o3d
import numpy as np
from spconv.pytorch.utils import PointToVoxel
from lib.utils import to_o3d_pcd, to_o3d_feats, to_array
import spconv.pytorch as spconv
from dataset.dataloader import collate_spconv_pair_fn
import torch
import copy
from model.resunet_spconv import FCGF_spconv

def spconv_vox(src_pcd, tgt_pcd, voxel_size):
    src_xyzmin, src_xyzmax = np.floor(np.percentile(src_pcd, 0, axis=0)), np.ceil(np.percentile(src_pcd, 100, axis=0))
    tgt_xyzmin, tgt_xyzmax = np.floor(np.percentile(tgt_pcd, 0, axis=0)), np.ceil(np.percentile(tgt_pcd, 100, axis=0))

    src_shape = (src_xyzmax - src_xyzmin) // voxel_size
    tgt_shape = (tgt_xyzmax - tgt_xyzmin) // voxel_size

    src_voxel_generator = PointToVoxel(vsize_xyz=[voxel_size] * 3,
                                       coors_range_xyz=[src_xyzmin[0], src_xyzmin[1], src_xyzmin[2], src_xyzmax[0],
                                                        src_xyzmax[1], src_xyzmax[2]],
                                       num_point_features=3,
                                       max_num_voxels=500000,
                                       max_num_points_per_voxel=1)
    tgt_voxel_generator = PointToVoxel(vsize_xyz=[voxel_size] * 3,
                                       coors_range_xyz=[tgt_xyzmin[0], tgt_xyzmin[1], tgt_xyzmin[2], tgt_xyzmax[0],
                                                        tgt_xyzmax[1], tgt_xyzmax[2]],
                                       num_point_features=3,
                                       max_num_voxels=500000,
                                       max_num_points_per_voxel=1)

    src_voxels_tv, src_indices_tv, _ = src_voxel_generator(torch.from_numpy(src_pcd).contiguous())
    tgt_voxels_tv, tgt_indices_tv, _ = tgt_voxel_generator(torch.from_numpy(tgt_pcd).contiguous())
    src_voxels_pts, src_voxels_coords = src_voxels_tv.numpy().squeeze(1), src_indices_tv.numpy()
    tgt_voxels_pts, tgt_voxels_coords = tgt_voxels_tv.numpy().squeeze(1), tgt_indices_tv.numpy()
    src_voxels_coords = src_voxels_coords[:, [2, 1, 0]]  # ZYX -> XYZ
    tgt_voxels_coords = tgt_voxels_coords[:, [2, 1, 0]]

    src_xyz, tgt_xyz = torch.from_numpy(src_voxels_pts).float(), torch.from_numpy(tgt_voxels_pts).float()

    ## batch index
    src_coords = torch.cat((torch.zeros((len(src_voxels_coords), 1)), torch.from_numpy(src_voxels_coords)), dim=-1)
    tgt_coords = torch.cat((torch.zeros((len(tgt_voxels_coords), 1)), torch.from_numpy(tgt_voxels_coords)), dim=-1)

    return src_xyz, tgt_xyz, src_voxels_coords, tgt_voxels_coords, src_shape, tgt_shape

def visualize_registration(src_ply, tgt_ply, pred_trans):
    src_ply.paint_uniform_color([1, 0, 0])
    tgt_ply.paint_uniform_color([0, 1, 0])
    src_ply_wrapped = copy.deepcopy(src_ply)
    src_ply_wrapped.transform(pred_trans)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=20)
    vis1.add_geometry(src_ply)
    vis1.add_geometry(tgt_ply)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Reg', width=960, height=540, left=960, top=20)
    vis2.add_geometry(src_ply_wrapped)
    vis2.add_geometry(tgt_ply)

    while True:
        vis1.update_geometry(src_ply)
        vis1.update_geometry(tgt_ply)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_ply_wrapped)
        vis2.update_geometry(tgt_ply)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    del vis1
    del vis2

if __name__ == '__main__':
    src_path = './misc/base_0.ply'
    tgt_path = './misc/base_1.ply'
    voxel_size = 1.0
    n_sample = 5000

    src_ply = o3d.io.read_point_cloud(src_path)
    tgt_ply = o3d.io.read_point_cloud(tgt_path)
    src_pcd = np.asarray(src_ply.points).astype(np.float32)
    tgt_pcd = np.asarray(tgt_ply.points).astype(np.float32)

    src_xyz, tgt_xyz, src_coords, tgt_coords, src_shape, tgt_shape = spconv_vox(src_pcd, tgt_pcd, voxel_size)
    src_features = torch.ones((len(src_coords), 1), dtype=torch.float32)
    tgt_features = torch.ones((len(tgt_coords), 1), dtype=torch.float32)
    list_data = [(src_xyz, tgt_xyz, src_coords, tgt_coords, src_features, tgt_features, torch.ones(1,2),
                  np.eye(4), src_shape, tgt_shape, None, np.ones((6, 6)))]

    ## init model
    model = FCGF_spconv()
    checkpoint = torch.load('ckpt_30.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.cuda()

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
    src_pcd = input_dict['pcd_src']
    tgt_pcd = input_dict['pcd_tgt']
    src_feats = out_src.features
    tgt_feats = out_tgt.features

    ## register
    src_sel = np.random.choice(len(src_pcd), min(len(src_pcd), n_sample), replace=False)
    tgt_sel = np.random.choice(len(tgt_pcd), min(len(tgt_pcd), n_sample), replace=False)

    src_pcd = src_pcd[src_sel]
    tgt_pcd = tgt_pcd[tgt_sel]
    src_feats = src_feats[src_sel]
    tgt_feats = tgt_feats[tgt_sel]

    result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
        to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), to_o3d_feats(src_feats), to_o3d_feats(tgt_feats),
        voxel_size * 1.5,
        o3d.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    pred_trans = result_ransac.transformation

    visualize_registration(src_ply, tgt_ply, pred_trans)


