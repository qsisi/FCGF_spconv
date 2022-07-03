import torch
from dataset.ThreedMatch import ThreeDMatchDataset, collate_spconv_pair_fn



def get_dataloader(infos, config, data_augmentation=True):
    dataset = ThreeDMatchDataset(infos=infos, config=config, data_augmentation=data_augmentation)
    shuffle = False if config.phase == 'test' else True
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,  # fix batch size to 1
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              num_workers=config.num_workers,
                                              collate_fn=collate_spconv_pair_fn,
                                              drop_last=False)

    return data_loader