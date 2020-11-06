import torchvision
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


class RandomCropToScale(object):

    def __init__(self, crop_ratio):
        assert isinstance(crop_ratio, float)
        self.crop_ratio = crop_ratio

    def __call__(self, sample):
        h = sample.shape[1]
        w = sample.shape[2]
        min_dim = min(h, w)
        new_dim = int(min_dim * self.crop_ratio)
        top = np.random.randint(0, h - new_dim)
        left = np.random.randint(0, w - new_dim)
        sample = sample[:, top: top+new_dim, left: left+new_dim]
        return sample


transform = torchvision.transforms.Compose([
                                RandomCropToScale(0.8),
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.Resize((224, 224)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])