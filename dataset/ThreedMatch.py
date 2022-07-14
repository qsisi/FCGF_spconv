import torch
from collections import defaultdict
from torch.utils.data import Dataset
import os
import open3d as o3d
import glob
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm
from lib.utils import read_ply, to_tsfm, to_o3d_pcd, to_tensor, get_correspondences
from lib.benchmark_util import load_log
from spconv.pytorch.utils import PointToVoxel

DATA_FILES = {
      'train': './configs/train_3dmatch.txt',
      'val': './configs/val_3dmatch.txt',
      'test': './configs/test_3dmatch.txt'
  }

class ThreeDMatchDataset(Dataset):
    def __init__(self, config, split, data_augmentation=True):
        super(ThreeDMatchDataset, self).__init__()
        self.base_dir = config.root
        self.data_augmentation = data_augmentation
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.search_radius
        self.rot_factor = config.rot_factor
        self.trans_scale = config.trans_scale
        self.jitter_noise = config.jitter_noise

        subset_names = open(DATA_FILES[split]).read().split()
        self.files = []
        for name in subset_names:
            fname = name + "*%.2f.txt" % 0.30
            fnames_txt = glob.glob(self.base_dir + "/" + fname)
            assert len(fnames_txt) > 0, f"Make sure that the path {self.base_dir} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # get pointcloud
        src_path = os.path.join(self.base_dir, self.files[item][0])
        tgt_path = os.path.join(self.base_dir, self.files[item][1])
        src_pcd = np.load(src_path)['pcd']
        tgt_pcd = np.load(tgt_path)['pcd']

        if self.data_augmentation:
            euler_src = np.random.rand(3) * np.pi * 2 / self.rot_factor      # [0, pi/2]
            euler_tgt = np.random.rand(3) * np.pi * 2 / self.rot_factor
            trans_src = -self.trans_scale + np.random.rand(3) * (2 * self.trans_scale)
            trans_tgt = -self.trans_scale + np.random.rand(3) * (2 * self.trans_scale)
            rot_src = Rotation.from_euler('zyx', euler_src).as_matrix()
            rot_tgt = Rotation.from_euler('zyx', euler_tgt).as_matrix()
            T0 = to_tsfm(rot_src, trans_src)
            T1 = to_tsfm(rot_tgt, trans_tgt)
            src_pcd = (rot_src @ src_pcd.T + trans_src[:, None]).T
            tgt_pcd = (rot_tgt @ tgt_pcd.T + trans_tgt[:, None]).T
            tsfm = T1 @ np.linalg.inv(T0)
            ### random jitter
            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.jitter_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.jitter_noise
        else:
            tsfm = np.eye(4)

        # build sparse tensor
        src_xyzmin, src_xyzmax = np.floor(np.percentile(src_pcd, 0, axis=0)), np.ceil(np.percentile(src_pcd, 100, axis=0))
        tgt_xyzmin, tgt_xyzmax = np.floor(np.percentile(tgt_pcd, 0, axis=0)), np.ceil(np.percentile(tgt_pcd, 100, axis=0))
        src_shape = (src_xyzmax - src_xyzmin) // self.voxel_size
        tgt_shape = (tgt_xyzmax - tgt_xyzmin) // self.voxel_size

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
        src_voxels_coords = src_voxels_coords[:, [2, 1, 0]]  # ZYX -> XYZ
        tgt_voxels_coords = tgt_voxels_coords[:, [2, 1, 0]]
        src_features = torch.ones((len(src_voxels_coords), 1), dtype=torch.float32)
        tgt_features = torch.ones((len(tgt_voxels_coords), 1), dtype=torch.float32)

        #### get correspondence ###
        matching_inds = get_correspondences(to_o3d_pcd(src_voxels_pts), to_o3d_pcd(tgt_voxels_pts), tsfm, self.search_voxel_size)

        src_xyz, tgt_xyz = to_tensor(src_voxels_pts).float(), to_tensor(tgt_voxels_pts).float()

        return src_xyz, tgt_xyz, src_voxels_coords, tgt_voxels_coords, src_features, tgt_features, \
               matching_inds, tsfm, src_shape, tgt_shape, None, np.ones((6, 6))

class ThreeDMatchTestset(Dataset):
    def __init__(self, config, split, data_augmentation):
        super(ThreeDMatchTestset, self).__init__()
        self.root = config.root
        self.voxel_size = config.voxel_size
        self.gt_logs = {}
        self.gt_infos = {}
        self.scene_frags = {}
        for scene in self.scene_list:
            log_path = os.path.join(config.root, scene + '-evaluation', 'gt.log')
            info_path = os.path.join(config.root, scene + '-evaluation', 'gt.info')
            self.gt_logs[scene] = load_log(log_path, stride=5)
            self.gt_infos[scene] = load_log(info_path, stride=7)
            self.scene_frags[scene] = len(self.gt_logs[scene])

        self.generate_test_info()

    @property
    def scene_list(self):
        return ['7-scenes-redkitchen',
                'sun3d-home_at-home_at_scan1_2013_jan_1',
                'sun3d-home_md-home_md_scan9_2012_sep_30',
                'sun3d-hotel_uc-scan3',
                'sun3d-hotel_umd-maryland_hotel1',
                'sun3d-hotel_umd-maryland_hotel3',
                'sun3d-mit_76_studyroom-76-1studyroom2',
                'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
            ]

    def generate_test_info(self):
        self.data_list = []
        for scene in self.scene_list:
            for id1_id2, gt_trans in self.gt_logs[scene].items():
                name = f'{scene}@{id1_id2}'
                self.data_list.append(name)

    def __getitem__(self, item):
        scene, id1, id2 = self.data_list[item].split('@')
        src_path = os.path.join(self.root, scene, f'cloud_bin_{id1}.ply')
        tgt_path = os.path.join(self.root, scene, f'cloud_bin_{id2}.ply')
        # src_ply = o3d.io.read_point_cloud(src_path)  # sometimes could not read *.ply silently, clustering screen
        # tgt_ply = o3d.io.read_point_cloud(tgt_path)
        # src_pcd = np.array(src_ply.points)
        # tgt_pcd = np.array(tgt_ply.points)
        ## uncomment the following two lines if *.ply could not be read silently which cluster screen
        src_pcd = read_ply(src_path)
        tgt_pcd = read_ply(tgt_path)

        if self.do_rotated:
            rot_int = 2 * np.pi
            R1 = Rotation.from_euler('zyx', [random.uniform(0, rot_int),
                                             random.uniform(0, rot_int),
                                             random.uniform(0, rot_int)]).as_matrix()

            R2 = Rotation.from_euler('zyx', [random.uniform(0, rot_int),
                                             random.uniform(0, rot_int),
                                             random.uniform(0, rot_int)]).as_matrix()
            src_pcd = (R1 @ src_pcd.T).T
            tgt_pcd = (R2 @ tgt_pcd.T).T

        tsfm = self.gt_logs[scene][f'{int(id1)}@{int(id2)}']
        info = self.gt_infos[scene][f'{int(id1)}@{int(id2)}']

        # build sparse tensor
        src_xyzmin, src_xyzmax = np.floor(np.percentile(src_pcd, 0, axis=0)), np.ceil(np.percentile(src_pcd, 100, axis=0))
        tgt_xyzmin, tgt_xyzmax = np.floor(np.percentile(tgt_pcd, 0, axis=0)), np.ceil(np.percentile(tgt_pcd, 100, axis=0))

        src_shape = (src_xyzmax - src_xyzmin) // self.voxel_size
        tgt_shape = (tgt_xyzmax - tgt_xyzmin) // self.voxel_size

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
        src_voxels_coords = src_voxels_coords[:, [2, 1, 0]]  # ZYX -> XYZ
        tgt_voxels_coords = tgt_voxels_coords[:, [2, 1, 0]]
        src_features = torch.ones((len(src_voxels_coords), 1), dtype=torch.float32)
        tgt_features = torch.ones((len(tgt_voxels_coords), 1), dtype=torch.float32)

        if self.do_rotated:
            src_voxels_pts = (R1.T @ src_voxels_pts.T).T
            tgt_voxels_pts = (R2.T @ tgt_voxels_pts.T).T

        src_xyz, tgt_xyz = to_tensor(src_voxels_pts).float(), to_tensor(tgt_voxels_pts).float()

        return src_xyz, tgt_xyz, src_voxels_coords, tgt_voxels_coords, src_features, tgt_features, \
               torch.ones(1,2), np.linalg.inv(tsfm), src_shape, tgt_shape, self.data_list[item], info

    def __len__(self):
        return sum([v for _, v in self.scene_frags.items()])

import argparse
import pdb
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/ghn/data/FCGF_data/threedmatch')
    parser.add_argument('--voxel_size', type=float, default=0.025)
    parser.add_argument('--search_radius', type=float, default=0.0375)
    parser.add_argument('--augment_noise', type=float, default=0.005)
    parser.add_argument('--vox_bnds', type=list, default=[-3.6, -2.4, 1.14])
    args = parser.parse_args()
    dataset = ThreeDMatchDataset(config=args, split='val')
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=collate_spconv_pair_fn,
                                              drop_last=False)
    # rand = torch.rand((1000,1)).cuda()
    all_mins = np.empty((0, 3))
    data_loader_iter = data_loader.__iter__()
    for i in tqdm(range(data_loader.dataset.__len__())):
        input_dict = data_loader_iter.next()[0]
        all_mins = np.concatenate([all_mins, input_dict], axis=0)
    print(all_mins.max(0))
