import pickle
import numpy as np
import open3d as o3d
import torch
import copy
import pdb
from plyfile import PlyData,PlyElement

def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    x_data = np.array(pc['x'])
    y_data = np.array(pc['y'])
    z_data = np.array(pc['z'])
    pc_array = np.hstack([x_data[:, None], y_data[:, None], z_data[:, None]])
    return pc_array

def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def get_correspondences(src, tgt, trans, search_voxel_size, K=None):
    src_pcd = copy.deepcopy(src)
    tgt_pcd = copy.deepcopy(tgt)
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)

    return correspondences