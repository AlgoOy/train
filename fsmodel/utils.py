import os
import torch
import numpy as np

from . import config as fs_config


def get_meta_batch(meta_batch_size, data_loader, iterator, channel=3, h=84, w=84):
    data = []
    for _ in range(meta_batch_size):
        try:
            #  "task_data" tensor is representing \
            #    the data of a task, with size of \
            #    [n_way, k_shot+q_query, 3, 84, 84]
            task_data = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            task_data = next(iterator)

        train_data = task_data[:, :fs_config.k_shot].reshape(-1, channel, h, w)
        val_data = task_data[:, fs_config.k_shot:].reshape(-1, channel, h, w)
        task_data = torch.cat((train_data, val_data), 0)
        data.append(task_data)

    return torch.stack(data).to(fs_config.device), iterator


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


def calculate_accuracy(logits, labels):
    """utility function for accuracy calculation"""
    acc = np.asarray(
        [(torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy())]
    ).mean()
    return acc


def calculate_model_size(model, file_name='model.pth', unit='KB'):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    param_size = next(model.parameters()).element_size()

    torch.save(model.state_dict(), file_name)
    file_size = os.path.getsize(file_name)
    if unit == 'KB':
        print(f'Params Size: {num_params * param_size / 1024:.2f} KB,'
              f' File Size: {file_size / 1024:.2f} KB')
    elif unit == 'MB':
        print(f'Params Size: {num_params * param_size / (1024 * 1024):.2f} KB,'
              f' File Size: {file_size / (1024 * 1024):.2f} MB')
    else:
        raise NotImplementedError
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        raise FileNotFoundError
