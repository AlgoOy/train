import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


# Dataset for few shot IDS
class FewShotIDS(Dataset):
    def __init__(self, file_path, k_shot, q_query):
        """
        file_path: 数据集路径
        split: 0.7 表示 70% 的任务用于训练（支持集+查询集），30% 用于验证
        """
        # 遍历文件，recursive=True允许使用**匹配所有文件，*表示匹配零个或多个字符
        self.file_list = [f for f in glob.glob(file_path + "*/*/", recursive=True)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_shot + q_query
        self.samples = len(os.listdir(self.file_list[0]))

        random.shuffle(self.file_list)

    def __getitem__(self, idx):
        sample = np.arange(self.samples)
        np.random.shuffle(sample)

        img_path = self.file_list[idx]
        img_list = [f for f in glob.glob(img_path + "/*.png", recursive=True)]
        img_list.sort()

        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        imgs = torch.stack(imgs)[sample[: self.n]]
        return imgs

    def __len__(self):
        return len(self.file_list)
