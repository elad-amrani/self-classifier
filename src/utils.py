import glob
import os
import numpy as np
import torch as th
import random

from torch.utils.data import Dataset
from PIL import Image, ImageFilter


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def set_bn_eval(module):
    if isinstance(module, th.nn.modules.batchnorm._BatchNorm):
        module.eval()


class SscDataset(Dataset):
    def __init__(self, root_dir, transform=None, size=-1):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = np.array(glob.glob(os.path.join(self.root_dir, '*/*')))
        self.size = len(self.img_paths) if size == -1 else size

        # shuffle dataset
        np.random.seed(0)
        self.img_paths = np.random.choice(self.img_paths, size=self.size, replace=False)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_paths[idx]
        original_image = pil_loader(img_name)

        if self.transform:
            view1 = self.transform(original_image)
            view2 = self.transform(original_image)

        return view1, view2, idx


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def set_lsf_env(world_size):
    ngpus_per_node = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    local_rank = int(os.environ.get('LSF_PM_XPROCID', 1)) - 1
    node_rank = int(os.environ.get('LSF_PM_XMACHID', 1)) - 1
    rank = node_rank * ngpus_per_node + local_rank
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_PORT'] = '12345'
    os.environ['MASTER_ADDR'] = os.environ.get('LSF_FROM_HOST', 'localhost')

    return rank, local_rank


class PrintMultiple(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()
