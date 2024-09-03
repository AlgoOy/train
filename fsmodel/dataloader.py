import torch
from torch.utils.data import DataLoader

from .dataset import FewShotIDS
from . import config as fs_config


def dataloader_init(train_ratio=0.6, val_ratio=0.2, shuffle=True, num_workers=0):
    dataset = FewShotIDS(fs_config.dataset_path, fs_config.k_shot, fs_config.q_query)
    train_num = int(train_ratio * len(dataset))
    val_num = int(val_ratio * len(dataset))
    test_num = len(dataset) - train_num - val_num
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_num, val_num, test_num]
    )
    train_loader = DataLoader(train_set, batch_size=fs_config.n_way,
                              num_workers=num_workers, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=fs_config.n_way,
                            num_workers=num_workers, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=fs_config.n_way,
                             num_workers=num_workers, shuffle=shuffle, drop_last=True)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)
    return (train_loader, val_loader, test_loader), (train_iter, val_iter, test_iter)
