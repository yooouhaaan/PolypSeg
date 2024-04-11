import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

# Real Annotation number :    0,  110, 120, 130, 140, 160, 190
# Mapping Annotation number : 0,  1,   2,   3,   4,   5,   6

IGNORE_INDEX = 255
ann_to_idx = {0:IGNORE_INDEX,
              110:0,
              120:1,
              130:2,
              140:3,
              160:4,
              190:5,
              255:IGNORE_INDEX}

class ForestImageSegmentationDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', transform=None, target_transform=None):
        super(ForestImageSegmentationDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.dataframe = pd.read_csv(os.path.join(dataset_dir, '{}_dataframe.csv'.format(mode)))#[:10]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.mode, self.dataframe['image_path'][idx])
        label_path = os.path.join(self.dataset_dir, self.mode, self.dataframe['label_path'][idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        label = np.array(label, dtype=np.uint8)
        for key, value in ann_to_idx.items():
            label[label == key] = value

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label) * 255

        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

# if __name__=='__main__':
#     import os
#
#     import natsort
#     import pandas as pd
#     from glob import glob
#     from tqdm import tqdm
#
#     root_dir = '/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection/Forest_Type_Segmentation'
#     print(os.getcwd())
#
#     train_image_path_list = natsort.natsorted(glob(os.path.join(root_dir, 'train', 'image', 'TS2_Fine', 'AP_IMAGE_512', '*.tif')))
#     train_label_path_list = natsort.natsorted(glob(os.path.join(root_dir, 'train', 'label', 'TL2_Fine', 'AP_512', 'FGT_TIF', '*.tif')))
#
#     print("len(train_image_path_list) : ", len(train_image_path_list))
#     print("len(train_label_path_list) : ", len(train_label_path_list))
#
#     print("train_image_path_list[:10] : ", train_image_path_list[:10])
#     print("train_label_path_list[:10] : ", train_label_path_list[:10])
#
#     train_dataframe = pd.DataFrame(columns=['image_path', 'label_path'])
#     print(train_dataframe)
#     for idx, (image_path, label_path) in tqdm(enumerate(zip(train_image_path_list, train_label_path_list))):
#         train_dataframe.loc[idx] = ['/'.join(image_path.split('/')[12:]), '/'.join(label_path.split('/')[12:])]
#     print(train_dataframe)
#     train_dataframe.to_csv(os.path.join(root_dir, 'train_dataframe.csv'), index=False)
#
#     val_image_path_list = natsort.natsorted(glob(os.path.join(root_dir, 'val', 'image', 'VS2_Fine', 'AP_IMAGE_512', '*.tif')))
#     val_label_path_list = natsort.natsorted(glob(os.path.join(root_dir, 'val', 'label', 'VL2_Fine', 'AP_512', 'FGT_TIF', '*.tif')))
#
#     print("len(val_image_path_list) : ", len(val_image_path_list))
#     print("len(val_label_path_list) : ", len(val_label_path_list))
#
#     print("val_image_path_list[:10] : ", val_image_path_list[:10])
#     print("val_label_path_list[:10] : ", val_label_path_list[:10])
#
#     val_dataframe = pd.DataFrame(columns=['image_path', 'label_path'])
#     print(val_dataframe)
#     for idx, (image_path, label_path) in tqdm(enumerate(zip(val_image_path_list, val_label_path_list))):
#         val_dataframe.loc[idx] = ['/'.join(image_path.split('/')[12:]), '/'.join(label_path.split('/')[12:])]
#     print(val_dataframe)
#     val_dataframe.to_csv(os.path.join(root_dir, 'val_dataframe.csv'), index=False)
