import torch
from dataset.ThreedMatch import ThreeDMatchDataset, ThreeDMatchTestset
from lib.utils import to_tensor
import pdb

def collate_spconv_pair_fn(list_data):
    src_xyz, tgt_xyz, src_coords, tgt_coords, src_feats, tgt_feats, \
    matching_inds, tsfm, src_shape, tgt_shape, scene, info = list(zip(*list_data))
    src_coords, tgt_coords = src_coords[0], tgt_coords[0]
    src_shape, tgt_shape = to_tensor(src_shape[0]), to_tensor(tgt_shape[0])

    # prepare inputs
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
        'tsfm': to_tensor(tsfm[0]),
        'grid_size': torch.cat([src_shape, tgt_shape]),
        ### only used for test part ###
        'scene': scene[0],
        'info': to_tensor(info[0]),
    }

def get_dataloader(config, split='train'):
    assert split in ['train', 'val', 'test']
    data_augmentation = True if split in ['train'] else False
    shuffle = True if split in ['train'] else False
    if split in ['train', 'val']:
        dataset = ThreeDMatchDataset(config=config, split=split, data_augmentation=data_augmentation)
    else:
        dataset = ThreeDMatchTestset(config=config, split=split, data_augmentation=data_augmentation)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,  # fix batch size to 1
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              num_workers=config.num_workers,
                                              collate_fn=collate_spconv_pair_fn,
                                              drop_last=False)

    return data_loader