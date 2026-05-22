# ------------------------------------------------------------------------
# For the Dataset code, code was inspired by https://github.com/fvasluianu97/RLN2
# 
# Modifications made by Youngjin Oh
# ------------------------------------------------------------------------

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import glob
import cv2


def load_img_resize(img, size):
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#### Ambient Lighting Normalization (Color) ####
class ImageDataset_CL3AN(Dataset):
    def __init__(self, image_dirs='./datasets/'):
        self.image_dirs = image_dirs

        gt_target_dir = f"{self.image_dirs}GT"
        self.gt_paths = []
        
        scanned_files = os.listdir(gt_target_dir)
        for item in scanned_files:
            if item.endswith('.png'):
                self.gt_paths.append(item)
        
        self.in_paths = []
        search_query = f"{self.image_dirs}**"
        
        for path_str in glob.iglob(search_query, recursive=True):
            if path_str.endswith("_IN.png"):
                normalized_str = path_str.replace('\\', '/')
                self.in_paths.append(normalized_str)
        
        self.path_dicts = {}
        for current_in_path in self.in_paths:
            extracted_filename = os.path.basename(current_in_path)
            file_identifier = extracted_filename.split('_')[0]
            
            mapped_gt_path = f"{gt_target_dir}/{file_identifier}_GT.png"
            
            self.path_dicts[current_in_path] = mapped_gt_path

        self.in_files = list(self.path_dicts.keys())
        self.gt_files = list(self.path_dicts.values())

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        in_name = self.in_files[idx]
        gt_name = self.gt_files[idx]

        in_image = (cv2.cvtColor(cv2.imread(in_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)) / 255
        gt_image = (cv2.cvtColor(cv2.imread(gt_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)) / 255

        if self.transform:
            in_image = self.transform(in_image)
            gt_image = self.transform(gt_image)

        sample = in_image, gt_image

        return sample
