# -*- coding: utf-8 -*-
"""
@Time ： 2021/9/27 15:36
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：dataset.py
@IDE ：PyCharm

"""
from torch.utils.data import Dataset
from PIL import Image
import json
import os


class CytoDataset(Dataset):
    def __init__(self, root, transformers):
        self.root = root
        self.transformers = transformers
        self.all_image_files = self.parser_files()

    def parser_files(self):
        all_files = []
        for d_type in ['neg', 'pos']:
            for tt in ['test', 'train']:
                dir = os.path.join(self.root, d_type, tt)
                slides = os.listdir(dir)
                for slide in slides:
                    fp = os.path.join(self.root, d_type, tt, dir, slide)
                    images = os.listdir(fp)
                    all_files += [os.path.join(fp, i) for i in images]
        return all_files

    def __len__(self):
        return len(self.all_image_files)

    def __getitem__(self, item):
        path = self.all_image_files[item]
        img = Image.open(path)
        img = self.transformers(img)
        return img, 0




