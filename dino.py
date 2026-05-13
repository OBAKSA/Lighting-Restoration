import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


# DINOv2
class DINOv2(nn.Module):
    def __init__(self):
        super(DINOv2, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_size = 14

    def forward(self, img):
        # img = F.interpolate(img, [1120, 1680])
        img = (img - self.mean) / self.std
        x = self.backbone.forward_features(img)["x_norm_patchtokens"]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], img.shape[2] // self.patch_size, img.shape[3] // self.patch_size)
        x = F.normalize(x, dim=-3, p=2)

        layers = [1, 6]
        features = []
        fs = []
        y = img
        y = self.backbone.prepare_tokens_with_masks(y, None)
        for idx, block in enumerate(self.backbone.blocks):
            y = block(y)
            if idx in layers:
                features.append(y)
        for i, f in enumerate(features):
            f = f[:, self.backbone.num_register_tokens + 1:]
            f = f.permute(0, 2, 1)
            f = f.reshape(f.shape[0], f.shape[1], img.shape[2] // self.patch_size, img.shape[3] // self.patch_size)
            f = F.normalize(f, dim=-3, p=2)
            fs.append(f)

        out = torch.cat([x, fs[0], fs[1]], dim=1)

        return out
