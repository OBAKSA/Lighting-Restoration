import os
import shutil
import logging
import math
import numpy as np
import torch
import cv2
import glob
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


def refresh_folder(dir):
    """
    If directory does not exist, create.
    If directory exists, delete then create.
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)


class AverageMeter(object):
    """
    Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def logger(log_adr, logger_name, mode='w'):
    """
    Logger
    """
    # create logger
    _logger = logging.getLogger(logger_name)
    # set level
    _logger.setLevel(logging.INFO)
    # set format
    formatter = logging.Formatter('%(message)s')
    # stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    # file
    file_handler = logging.FileHandler(log_adr, mode=mode)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    return _logger


def date_time(secs):
    day = int(secs // (24 * 3600))
    secs = secs % (24 * 3600)
    hour = int(secs // 3600)
    secs %= 3600
    minutes = int(secs // 60)
    secs %= 60
    seconds = int(secs)
    return f'{day} d {hour} h {minutes} m {seconds} s'


def create_plateau_blending_mask(patch_height, patch_width, overlap_size_y, overlap_size_x, min_weight):
    """
    (NumPy Only)
    Generates a 2D blending mask that smoothly transitions from 1 at the center to 0 at the overlapping edges.
    Uses the Sigmoid function instead of SciPy's norm.cdf.

    Args:
        patch_height (int): The height of the entire patch (e.g., 384).
        patch_width (int): The width of the entire patch (e.g., 384).
        overlap_size_y (int): The pixel size of the overlapping region along the y-axis (e.g., 32).
        overlap_size_x (int): The pixel size of the overlapping region along the x-axis (e.g., 32).
        min_weight (float): The minimum value of the mask to prevent division by zero (e.g., 0.1).

    Returns:
        numpy.ndarray: A 2D blending mask of shape (patch_height, patch_width).
    """
    ramp_size_y = min(overlap_size_y, patch_height // 2)
    ramp_size_x = min(overlap_size_x, patch_width // 2)

    y = np.linspace(-6, 6, ramp_size_y)
    ramp_1d_y = 1 / (1 + np.exp(-y))
    x = np.linspace(-6, 6, ramp_size_x)
    ramp_1d_x = 1 / (1 + np.exp(-x))

    center_h = patch_height - 2 * ramp_size_y
    center_w = patch_width - 2 * ramp_size_x

    mask_y = np.concatenate([ramp_1d_y, np.ones(center_h), ramp_1d_y[::-1]])
    mask_x = np.concatenate([ramp_1d_x, np.ones(center_w), ramp_1d_x[::-1]])

    blending_mask = np.outer(mask_y, mask_x)
    scaled_mask = min_weight + blending_mask * (1 - min_weight)
    scaled_mask = torch.from_numpy(scaled_mask).float()

    return scaled_mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,h,w)


#####
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        self._last_lr = []
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            self._last_lr.append(lr)
