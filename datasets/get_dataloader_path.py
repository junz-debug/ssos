from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms
import os
import torchvision.transforms as transforms

class DataSet(Dataset):
    def __init__(self, root_path, domain,load_list,transform=None):
        self.root = f"{root_path}/{domain}"
        self.transform = transform
        label_name_list = os.listdir(self.root)
        self.label = []
        self.data = []
        print(f"Getting {domain} datasets")
        for index, label_name in enumerate(label_name_list):
            if label_name in load_list:
                continue
            images_list = os.listdir(f"{self.root}/{label_name}")
            for img_name in images_list:
                img_path = f"{self.root}/{label_name}/{img_name}"
                self.label.append(label_name_list.index(label_name))
                self.data.append(img_path)
        self.label = torch.tensor(self.label, dtype=torch.long)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.label[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
