import torch
from torch.utils.data import Dataset
import os
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm
from lib.utils import load_obj, to_tsfm, to_o3d_pcd, to_tensor, get_correspondences
import argparse
from spconv.pytorch.utils import PointToVoxel
import pdb

def collate_spconv_pair_fn(list_data):
    src_xyz, tgt_xyz, src_coords, tgt_coords, src_feats, tgt_feats, matching_inds, rot, trans = list(zip(*list_data))
    src_coords, tgt_coords = src_coords[0], tgt_coords[0]

    # prepare inputs for FCGF
    src_coords = torch.cat((torch.zeros((len(src_coords), 1)), torch.from_numpy(src_coords)), dim=-1)
    tgt_coords = torch.cat((torch.zeros((len(tgt_coords), 1)), torch.from_numpy(tgt_coords)), dim=-1)
    # concatenate xyz
    src_xyz = torch.cat(src_xyz, 0).float()
    tgt_xyz = torch.cat(tgt_xyz, 0).float()
    matching_inds_batch = torch.cat(matching_inds, 0).long()

    return {
        'pcd_src': src_xyz,
        'pcd_tgt': tgt_xyz,
        'src_C': src_coords,
        'src_F': src_feats[0],
        'tgt_C': tgt_coords,
        'tgt_F': tgt_feats[0],
        'correspondences': matching_inds_batch,
        'rot': rot[0],
        'trans': trans[0]
    }

class ThreeDMatchDataset(Dataset):
    def __init__(self, infos, config, data_augmentation=True):
        super(ThreeDMatchDataset, self).__init__()
        self.infos = infos
        self.base_dir = config.root
        self.data_augmentation = data_augmentation
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.search_radius

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self, item):
        # get transformation
        rot = self.infos['rot'][item]
        trans = self.infos['trans'][item]

        # get pointcloud
        src_path = os.path.join(self.base_dir, self.infos['src'][item])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        if (trans.ndim == 1):
            trans = trans[:, None]

        # build sparse tensor
        src_xyzmin, src_xyzmax = np.percentile(src_pcd, 0, axis=0), np.percentile(src_pcd, 100, axis=0)
        tgt_xyzmin, tgt_xyzmax = np.percentile(tgt_pcd, 0, axis=0), np.percentile(tgt_pcd, 100, axis=0)
        # pdb.set_trace()
        src_voxel_generator = PointToVoxel(vsize_xyz=[self.voxel_size] * 3,
                                            coors_range_xyz=[src_xyzmin[0], src_xyzmin[1], src_xyzmin[2], src_xyzmax[0],
                                                        src_xyzmax[1], src_xyzmax[2]],
                                            num_point_features=3,
                                            max_num_voxels=500000,
                                            max_num_points_per_voxel=1)
        tgt_voxel_generator = PointToVoxel(vsize_xyz=[self.voxel_size] * 3,
                                           coors_range_xyz=[tgt_xyzmin[0], tgt_xyzmin[1], tgt_xyzmin[2], tgt_xyzmax[0],
                                                            tgt_xyzmax[1], tgt_xyzmax[2]],
                                           num_point_features=3,
                                           max_num_voxels=500000,
                                           max_num_points_per_voxel=1)
        src_voxels_tv, src_indices_tv, _ = src_voxel_generator(torch.from_numpy(src_pcd).contiguous())
        tgt_voxels_tv, tgt_indices_tv, _ = tgt_voxel_generator(torch.from_numpy(tgt_pcd).contiguous())
        src_voxels_pts, src_voxels_coords = src_voxels_tv.numpy().squeeze(1), src_indices_tv.numpy()
        tgt_voxels_pts, tgt_voxels_coords = tgt_voxels_tv.numpy().squeeze(1), tgt_indices_tv.numpy()
       
        # SERIOUS WRONG WAY OF SWAPING HERE! #
        # src_voxels_coords[:, 0], src_voxels_coords[:, 2] = src_voxels_coords[:, 2], src_voxels_coords[:, 0]
        # tgt_voxels_coords[:, 0], tgt_voxels_coords[:, 2] = tgt_voxels_coords[:, 2], tgt_voxels_coords[:, 0]
        src_voxels_coords = src_voxels_coords[:, [2, 1, 0]]
        tgt_voxels_coords = tgt_voxels_coords[:, [2, 1, 0]]
        src_features = torch.ones((len(src_voxels_coords), 1), dtype=torch.float32)
        tgt_features = torch.ones((len(tgt_voxels_coords), 1), dtype=torch.float32)

        ### quick visualization ###
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(src_pcd)
        # o3d.io.write_point_cloud('src.ply', pcd)
        # pcd.points = o3d.utility.Vector3dVector(src_voxels_coords)
        # o3d.io.write_point_cloud('vox.ply', pcd)
        # pcd.points = o3d.utility.Vector3dVector(src_voxels_pts)
        # o3d.io.write_point_cloud('vox_pts.ply', pcd)
        # pdb.set_trace()
        #### get correspondence ###
        tsfm = to_tsfm(rot, trans)
        matching_inds = get_correspondences(to_o3d_pcd(src_voxels_pts), to_o3d_pcd(tgt_voxels_pts), tsfm, self.search_voxel_size)

        src_xyz, tgt_xyz = to_tensor(src_voxels_pts).float(), to_tensor(tgt_voxels_pts).float()
        rot, trans = to_tensor(rot), to_tensor(trans)

        # pdb.set_trace()
        return src_xyz, tgt_xyz, src_voxels_coords, tgt_voxels_coords, src_features, tgt_features, matching_inds, rot, trans

if __name__ == '__main__':
    info_train = load_obj('/home/haining/PredatorMetric/configs/indoor/train_info.pkl')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/ssd1/ghn_data/PredatorDataset')
    parser.add_argument('--voxel_size', type=float, default=0.025)
    parser.add_argument('--search_radius', type=float, default=0.0375)
    parser.add_argument('--augment_noise', type=float, default=0.005)
    args = parser.parse_args()
    dataset = ThreeDMatchDataset(infos=info_train, config=args, data_augmentation=True)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=collate_spconv_pair_fn,
                                              drop_last=False)
    rand = torch.rand((1000,1)).cuda()
    
    data_loader_iter = data_loader.__iter__()
    for i in tqdm(range(data_loader.dataset.__len__())):
        input_dict = data_loader_iter.next()
        # pdb.set_trace()
