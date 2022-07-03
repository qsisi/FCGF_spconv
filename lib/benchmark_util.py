import pdb
import torch
import numpy as np
import nibabel.quaternions as nq
from lib.utils import to_o3d_pcd, to_o3d_feats

def load_log(log_path, stride):
    with open(log_path, 'r') as f:
        content = f.readlines()
    assert len(content) % stride == 0, "wrong stride number"
    frags = len(content) // stride
    logs = {}
    for frag in range(frags):
        id1, id2, _ = content[stride * frag].strip().split('\t')
        matrix = []
        for increment in range(1, stride):
            matrix.append(np.array(content[stride * frag + increment].strip().split('\t')).astype(np.float32))
        T = np.vstack(matrix)
        logs[f'{int(id1)}@{int(id2)}'] = T

    assert len(logs.keys()) == frags

    return logs

def get_corr_from_dist_matrix(matrix, mutual=True):
    row_mask = matrix == matrix.min(dim=1, keepdim=True)[0]

    col_mask = matrix == matrix.min(dim=0, keepdim=True)[0]

    if mutual:
        mask = torch.logical_and(row_mask, col_mask)
    else:
        mask = torch.logical_or(row_mask, col_mask)

    return mask.nonzero()

def computeTransformationErr(trans, info):
    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]
    return p.item()



