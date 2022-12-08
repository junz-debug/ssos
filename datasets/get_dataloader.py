from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms
import os
import torchvision.transforms as transforms

class my_DataSet(Dataset):
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
                img = Image.open(f"{self.root}/{label_name}/{img_name}").convert('RGB')
                img = np.array(img)
                self.label.append(label_name_list.index(label_name))
                if self.transform is not None:
                    img = self.transform(img)
                self.data.append(img)
        self.data = torch.stack(self.data)
        self.label = torch.tensor(self.label, dtype=torch.long)

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        return img, target

    def __len__(self):
        return len(self.data)

def my_DataLoader(root_path,domain,know_list,batch_size = 32,num_workers = 0,transform = None):
    data_set = my_DataSet(root_path, domain,know_list,transform)
    data_loader=DataLoader(data_set, batch_size=batch_size, shuffle = True, num_workers = num_workers, pin_memory = False)
    return data_loader,data_set




