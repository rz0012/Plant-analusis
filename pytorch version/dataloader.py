import torch
from torch.utils.data import Dataset
import numpy as np
import os


class dataset(Dataset):
    def __init__(self, list_id, label, data_dir, transform=None):
        self.label = label
        self.list_id = list_id
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index):
        try:
            data_id = self.list_id[index]
            X = torch.load(os.path.join(self.data_dir, data_id))
            if self.transform:
                X = self.transform(X)
            X = X.view(3,224,224)
            y = self.label[data_id]

            return X, y

        except Exception as e:
            pass 



        

