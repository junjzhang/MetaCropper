import torch
import pathlib

import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
import torch.utils.data.Dataset as Dataset
import PIL.Image as Image

IMAGE_NET_MEAN = [0.485, 456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class FATDataset(Dataset):
    def __init__(self, root_path, wrap_size):
        super().__init__()
        self.root_path = root_path

        normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        self.img_loader = transforms.Compose([
            transforms.Resize(wrap_size),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
