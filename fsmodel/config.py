import random
import numpy as np
import torch


# todo: 使用toml实现配置文件

class Config:
    def __init__(self):
        # 选择设备
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = "cuda:1"
            else:
                self.device = "cuda"
        else:
            self.device = "cpu"

        # 固定随机种子
        random_seed = 0
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        # 配置参数
        self.random_seed = random_seed
        self.n_way = 5
        self.k_shot = 5
        self.q_query = 15
        self.train_inner_train_step = 3
        self.val_inner_train_step = 3
        self.test_inner_train_step = 3
        self.inner_lr = 0.1
        self.meta_lr = 0.001
        self.meta_batch_size = 6
        self.max_epoch = 5000
        self.eval_batches = 2
        self.test_batches = 1
        self.dataset_path = "./Dataset/R-FSIDS_84_84_RGB/"

    def set_config(self, dataset_path, train_inner_train_step=3, val_inner_train_step=3, 
                   test_inner_train_step=3, inner_lr=0.1, meta_lr=0.001, meta_batch_size=6,
                   max_epoch=5000, eval_batches=2, test_batches=1):
        self.train_inner_train_step = train_inner_train_step
        self.val_inner_train_step = val_inner_train_step
        self.test_inner_train_step = test_inner_train_step
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_batch_size = meta_batch_size
        self.max_epoch = max_epoch
        self.eval_batches = eval_batches
        self.test_batches = test_batches
        self.dataset_path = dataset_path
