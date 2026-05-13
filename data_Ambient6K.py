from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
import cv2


#### Ambient Lighting Normalization (White) ####
class ImageDataset_Ambient6K(Dataset):
    def __init__(self, image_dirs='./datasets/'):
        self.image_dirs = image_dirs

        self.in_files = glob.glob(self.image_dirs + 'in/*.png')
        self.gt_files = glob.glob(self.image_dirs + 'gt/*.png')

        self.in_files.sort()
        self.gt_files.sort()

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
