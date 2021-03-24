import torchvision.transforms.functional as F
import torch
import pandas as pd

import torchvision.transforms as transforms
import PIL.Image as Image

from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset
from pathlib2 import Path
from einops import rearrange

IMAGE_NET_MEAN = [0.485, 456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

FILE_NAME = {'train': 'train.csv', 'test': 'test.csv'}


class LongEdgeResize():
    def __init__(self, wrap_size):
        self.wrap_size = wrap_size

    def __call__(self, img):
        size = self._calc_new_size(img)
        return F.resize(img, size)

    def _calc_new_size(self, img):
        size = img.size
        base_edge_idx = 0 if size[0] > size[1] else 1
        size[int(not base_edge_idx)] *= self.wrap_size / \
            base_edge_idx[base_edge_idx]
        size[base_edge_idx] = 256
        return list(map(round, size))


class FATDataset(Dataset):
    def __init__(self, root_path, wrap_size, task_type='train'):
        super().__init__()

        assert task_type in [
            'train', 'test'], 'No {} split, only train and test' % (task_type)

        normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

        self.img_loader = transforms.Compose([
            LongEdgeResize(wrap_size),
            transforms.ToTensor(),
            normalize,
        ])

        root_path = Path(root_path)
        self.pic_dir = root_path / 'PQ_Set'
        file_name = FILE_NAME[task_type]
        info_table = pd.read_csv(root_path/file_name)
        self.pic_name_list = info_table.iloc[:, 0]
        self.ratio_crop_tensor = info_table.iloc[:, 1:]

    def __len__(self):
        return self.pic_name_list.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.pic_dir/self.pic_name_list.iloc[idx])
        img_tensor = self.img_loader(img)
        ratio_crop_tensor = self.ratio_crop_tensor[idx]
        ratio_crop_tensor = ratio_crop_tensor[not torch.isnan(
            ratio_crop_tensor)]
        ratio_crop_tensor = rearrange(
            ratio_crop_tensor, 'b (r n) -> b r n', n=5)
        ratio = ratio_crop_tensor[:, 0]
        crop_pos = ratio_crop_tensor[:, 1:]
        return img, ratio, crop_pos
