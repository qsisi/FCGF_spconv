import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d

from lib.timer import Timer, AverageMeter

from model import load_model
from lib.benchmark_util import *

import torch
import pdb

scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

def extract_features_batch(model, source_path, target_path, voxel_size, device):
#   pdb.set_trace()
  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for fo in folders:
    if 'evaluation' in fo:
      continue
#     pdb.set_trace()
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file
      pcd = o3d.io.read_point_cloud(fi)
      save_fn = "%s_%03d" % (fo_base, i)
      print(f"{i} / {len(files)}: {save_fn}")

      timer.tic()
      xyz_down, feature = extract_features(
          model,
          xyz=np.array(pcd.points),
          rgb=None,
          normal=None,
          voxel_size=voxel_size,
          device=device,
          skip_check=True)
      t = timer.toc()
      if i > 0:
        tmeter.update(t)
        num_feat += len(xyz_down)

      np.savez_compressed(
          os.path.join(target_path, save_fn),
          points=np.array(pcd.points),
          xyz=xyz_down,
          feature=feature.detach().cpu().numpy())
      if i % 20 == 0 and i > 0:
        print(
            f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
        )

  f.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source', default=None, type=str, help='path to 3dmatch test dataset')
  parser.add_argument(
      '--source_high_res',
      default=None,
      type=str,
      help='path to high_resolution point cloud')
  parser.add_argument(
      '--target', default=None, type=str, help='path to produce generated data')
  parser.add_argument(
      '-m',
      '--model',
      default=None,
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.05,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument('--extract_features', action='store_true')
  parser.add_argument('--evaluate_fmr', action='store_true')
  parser.add_argument(
      '--evaluate_registration',
      action='store_true',
      help='The target directory must contain extracted features')
  parser.add_argument(
      '--num_rand_keypoints',
      type=int,
      default=5000,
      help='Number of random keypoints for each scene')
  parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
  parser.add_argument('--distance_threshold', default=0.10, type=float)

  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

  if args.extract_features:
    assert args.model is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model)
    num_feats = 1
    Model = load_model('ResUNet_spconv')
    model = Model()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    with torch.no_grad():
      extract_features_batch(model, args.source, args.target, args.voxel_size,
                             device)

  if args.evaluate_fmr:
    return_dict = Manager().dict()
#     jobs = []
    for scene in scene_list:
#         p = Process(target=register_one_scene, args=(args.inlier_ratio_threshold, args.distance_threshold, args.target, return_dict, scene))
        register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, args.target, args.num_rand_keypoints, return_dict, scene)
#         jobs.append(p)
#         p.start()

#     for proc in jobs:
#         proc.join()

    recalls = [v[0] for k, v in return_dict.items()]
    inlier_nums = [v[1] for k, v in return_dict.items()]
    inlier_ratios = [v[2] for k, v in return_dict.items()]
    print("*" * 40)
    print(recalls)
    print(f"All 8 scene, average recall: {np.mean(recalls):.2f}%")
    print(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    print(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios)*100:.2f}%")




  if args.evaluate_registration:
    assert (args.target is not None)
    with torch.no_grad():
      registration(args.target, args.voxel_size)