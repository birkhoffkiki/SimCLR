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
    def __init__(self, root, json_file, transformers):
        self.root = root
        self.transformers = transformers
        with open(json_file) as f:
            files = json.load(f)

        self.all_image_files = self.parser_files(files)

    def parser_files(self, files: dict):
        all_files = []
        for d_type, s in files.items():
            for slide, _ in s.items():
                dir = os.path.join(self.root, d_type, slide)
                img_files = os.listdir(dir)
                all_files += [os.path.join(dir, i) for i in img_files]
        return all_files

    def __len__(self):
        return len(self.all_image_files)

    def __getitem__(self, item):
        path = self.all_image_files[item]
        img = Image.open(path)
        for t in self.transformers:
            img = t(img)
        return img




