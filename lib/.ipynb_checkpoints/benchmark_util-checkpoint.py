import os
import numpy as np
from os import listdir
from os.path import isfile, isdir, join, splitext
import re
import torch
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel
from multiprocessing import Process, Manager
from lib.timer import Timer, AverageMeter
import open3d as o3d
import pdb

def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path, mode=0o755)
    
def build_correspondence(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """

    distance = np.sqrt(2 - 2 * (source_desc @ target_desc.T))
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    target_idx = np.argmin(distance, axis=0)
    target_dis = np.min(distance, axis=0)

    result = []
    for i in range(len(source_idx)):
        if target_idx[source_idx[i]] == i:
            result.append([i, source_idx[i]])
    return np.array(result)

def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result

def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, sample_points, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    gtpath = f'/data/ghn/FCGF_data/threedmatch_test/{scene}-evaluation/'
    pdb.set_trace()
    gtLog = loadlog(gtpath)
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"/data/ghn/FCGF_data/threedmatch_test/{scene}/"
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = str(id1).zfill(3)
            cloud_bin_t = str(id2).zfill(3)
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                data0 = np.load(os.path.join(save_path, f'{scene}_{cloud_bin_s}.npz'))
                data1 = np.load(os.path.join(save_path, f'{scene}_{cloud_bin_t}.npz'))
                
                source_keypts = data0['xyz']
                target_keypts = data1['xyz']
               
                source_desc = data0['feature']
                target_desc = data1['feature']
                
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)
                
                
                source_indices = np.arange(len(source_keypts))
                target_indices = np.arange(len(target_keypts))
                if len(source_keypts) > sample_points:
                    source_indices = np.random.choice(len(source_keypts), sample_points, replace=False)
                if len(target_keypts) > sample_points:
                    target_indices = np.random.choice(len(target_keypts), sample_points, replace=False)
                
                source_keypts = source_keypts[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                source_desc = source_desc[source_indices, :]
                target_desc = target_desc[target_indices, :]
                pdb.set_trace()     
                
                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)
    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    print(f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg
    
def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None
  if both rgb and normal are None, we use Nx1 one vector as an input
  if device is None, it tries to use gpu by default
  if skip_check is True, skip rigorous checks to speed up
  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  xyzmin, xyzmax = np.percentile(xyz, 0, axis=0), np.percentile(xyz, 100, axis=0)
  voxel_generator = PointToVoxel(vsize_xyz=[voxel_size] * 3,
                                            coors_range_xyz=[xyzmin[0], xyzmin[1], xyzmin[2], xyzmax[0],
                                                        xyzmax[1], xyzmax[2]],
                                            num_point_features=3,
                                            max_num_voxels=500000,
                                            max_num_points_per_voxel=1)
  voxels_tv, indices_tv, _ = voxel_generator(torch.from_numpy(xyz).contiguous())
  voxels_pts, voxels_coords = voxels_tv.numpy().squeeze(1), indices_tv.numpy()
  voxels_coords = voxels_coords[:, [2, 1, 0]]  
    
  feats = torch.ones((len(voxels_pts) ,1), dtype=torch.float32)
  coords = torch.tensor(voxels_coords, dtype=torch.int32)
  coords = torch.cat((torch.zeros((len(coords), 1)), coords), dim=-1)
  coords_min, coords_max = np.floor(np.percentile(coords, 0, axis=0)).astype(np.int32), \
                                             np.floor(np.percentile(coords, 100, axis=0)).astype(np.int32)
  stensor = spconv.SparseConvTensor(feats.to(device), coords.int().to(device), (coords_max - coords_min)[1:], batch_size=1)
#   pdb.set_trace()
  return voxels_pts, model(stensor).features

def get_folder_list(path):
  folder_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
  folder_list = sorted_alphanum(folder_list)
  return folder_list

def sorted_alphanum(file_list_ordered):

  def convert(text):
    return int(text) if text.isdigit() else text

  def alphanum_key(key):
    return [convert(c) for c in re.split('([0-9]+)', key)]

  return sorted(file_list_ordered, key=alphanum_key)

def get_file_list(path, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  else:
    file_list = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and splitext(f)[1] == extension
    ]
  file_list = sorted_alphanum(file_list)
  return file_list

