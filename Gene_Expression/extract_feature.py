import os
import timm
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from model import Pix2Feature

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(880310)

test_tfm = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])  

class My_Dataset(Dataset):
    def __init__(self, img_folder, transforms):
        self.img_folder = img_folder
        self.transforms = transforms

        self.slide_list = os.listdir(self.img_folder)

        self.patch_list = []
        for slide in self.slide_list:
            for patch in os.listdir(os.path.join(img_folder, slide)):
                self.patch_list.append(patch)

        print(f"{len(self.slide_list)} slides.")
        print(f"{len(self.patch_list)} patches.")

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch_name = self.patch_list[idx]
        slide_name = patch_name.split('_')[0]
        image = Image.open(os.path.join(self.img_folder, slide_name, patch_name))
        image = self.transforms(image)
        return slide_name, patch_name, image

ds = My_Dataset('/staging/biology/r10945019/data/Lung/TCGA_LUAD_40x/DX(no_matched_pathway)', test_tfm)
dl = DataLoader(ds, batch_size=32, num_workers=6, shuffle=False)

model = Pix2Feature(cnn_arch='inception_resnet_v2')
model = model.cuda()


model.eval()

for slide_name, patch_name, image in tqdm(dl):
    with torch.no_grad():
        feature = model(image.cuda())
        feature = feature.squeeze().cpu().detach().numpy()
    
    for i in range(feature.size(0)):
        if os.isdir(os.path.join('/staging/biology/r10945019/data/Lung/new_LUAD_Inception_Resnet_v2_feature', slide_name[i])) == False:
            os.mkdir(os.path.join('/staging/biology/r10945019/data/Lung/new_LUAD_Inception_Resnet_v2_feature', slide_name[i]))
        np.save(os.path.join('/staging/biology/r10945019/data/Lung/new_LUAD_Inception_Resnet_v2_feature', slide_name[i], patch_name[i]), feature[i, :])

'''
def extract_feature(loader, mode):
    model.eval()

    for slide_name, patch_name, image in loader:
        with torch.no_grad():
            feature = model(image.cuda())
            feature = feature.squeeze().cpu().detach().numpy()

        for i in range(feature.size(0)):
            if os.isdir(os.path.join(save_path, mode, slide_name[i])) == False:
                os.mkdir(os.path.join(save_path, mode, slide_name[i]))
            np.save(os.path.join(save_path, mode, slide_name[i], patch_name[i]), feature[i, :])
'''
#extract_feature(train_loader, 'train')
#extract_feature(val_loader, 'valid')
#extract_feature(test_loader, 'test')
