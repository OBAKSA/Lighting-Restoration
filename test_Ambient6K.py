import warnings

warnings.filterwarnings('ignore')

from time import time
import numpy as np
import imageio
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader

from utils import *
from pytorch_msssim import ssim
import lpips

from torch.utils.data import DataLoader
from data_Ambient6K import ImageDataset_Ambient6K

from model_DINOLight import DINOLight
from model_OmniLight import OmniLight

from dino import DINOv2

# Fix random seed for reproducibility
import random

random.seed(1994)
np.random.seed(1994)
torch.manual_seed(1994)
torch.cuda.manual_seed_all(1994)

device = torch.device('cuda')


def str2bool(s):
    return True if s.lower() == 'true' else False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default='1', help='number of images to be inferenced, fix as 1')
    parser.add_argument('--patch_batch_size', type=int, default='12',
                        help='evaluation number of patches, adjust this value based on your GPU memory capacity')
    parser.add_argument('--exp_name', type=str, default='omnilight',
                        help='the name of experiment')
    parser.add_argument('--ckpt_name', type=str, default='omnilight_bench.pth', help='the name of checkpoint file')
    parser.add_argument('--eval_path', type=str, default='./datasets/Ambient6K/',
                        help='evaluation file path')
    parser.add_argument('--save_gt_noisy', type=str2bool, default='False',
                        help='True for saving gt and noisy, else False')
    parser.add_argument('--foundation_type', type=str, default='dinov2', help='foundation model type')
    opt = parser.parse_args()

    # Make path to save results
    refresh_folder(f'./result/{opt.exp_name}/result_imgs')

    # Logger
    log = logger(f'./result/{opt.exp_name}/eval_log.txt', 'eval', 'w')
    opt_log = '-' * 15 + ' Options ' + '-' * 15 + '\n'
    for k, v in vars(opt).items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '-' * 39 + '\n'
    log.info(opt_log)

    # Dataset & Dataloader
    test_dataset = ImageDataset_Ambient6K(image_dirs=opt.eval_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    with torch.no_grad():
        # Restoration Model
        if opt.exp_name == 'dinolight':
            network = DINOLight(channels=32).cuda()
        elif opt.exp_name == 'omnilight':
            network = OmniLight(channels=32).cuda()
        else:
            network = OmniLight(channels=32).cuda()

        # Foundation Model
        if opt.foundation_type == 'dinov2':
            foundation_model = DINOv2().cuda()
        else:
            foundation_model = DINOv2().cuda()

        # Load checkpoint
        ckpt = torch.load(f'./pretrained_weights/{opt.exp_name}/{opt.ckpt_name}', map_location=torch.device('cuda'))
        state_dict = ckpt['network_state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        network.load_state_dict(new_state_dict)
        # Validate
        validate(opt, test_dataloader, network, foundation_model, log)


def validate(opt, test_dataloader, network, foundation_model, log):
    # Eval mode
    network.eval()
    foundation_model.eval()

    loss_fn = lpips.LPIPS(net='alex').to(device)

    # Validate
    psnr_avg = 0.
    ssim_avg = 0.
    lpips_avg = 0.
    num_data = 0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            in_image = data[0]
            in_image = in_image.to(device)
            gt_image = data[1]
            gt_image = gt_image.to(device)

            # restoration output
            patch_size = 448
            height = 960
            width = 1280
            overlap_y = 192
            overlap_x = 240

            restored = torch.zeros((1, 3, height, width)).to(device)
            weight_map = torch.zeros((1, 3, height, width)).to(device)

            # 1. Collect patches and their corresponding coordinates
            patches = []
            coords = []

            for y in range(0, height, patch_size - overlap_y):
                y1 = min(y, height - patch_size)
                y2 = y1 + patch_size
                for x in range(0, width, patch_size - overlap_x):
                    x1 = min(x, width - patch_size)
                    x2 = x1 + patch_size

                    # Crop patch and append to list (Shape: 1, 3, H, W)
                    patch = in_image[:, :, y1:y2, x1:x2]
                    patches.append(patch)
                    coords.append((y1, y2, x1, x2))

            # Concatenate all collected patches into a single batched tensor (Shape: N, 3, H, W)
            batched_patches = torch.cat(patches, dim=0).to(device)

            # Create blending mask once and reuse it to save memory and time
            weight = create_plateau_blending_mask(patch_height=patch_size, patch_width=patch_size,
                                                  overlap_size_y=overlap_y, overlap_size_x=overlap_x, min_weight=1e-6)
            weight = weight.to(device)

            # 2. Perform batched inference
            patch_batch_size = opt.patch_batch_size  # Adjust this value based on your GPU memory capacity
            restored_patches = []

            for i in range(0, batched_patches.size(0), patch_batch_size):
                # Python slicing safely handles the last batch even if it's smaller than patch_batch_size
                batch = batched_patches[i:i + patch_batch_size]

                feature = foundation_model(batch)
                if opt.exp_name == 'omnilight':
                    out_patch, _ = network(batch, feature)
                else:
                    out_patch = network(batch, feature)

                restored_patches.append(out_patch)

            # Concatenate the inferred mini-batches back into a single tensor (Shape: N, 3, H, W)
            restored_patches = torch.cat(restored_patches, dim=0)

            # 3. Reconstruct the final image using the original coordinates
            for i, (y1, y2, x1, x2) in enumerate(coords):
                # Extract the i-th restored patch (using [i:i+1] to maintain dimensions)
                r_patch = restored_patches[i:i + 1]

                restored[:, :, y1:y2, x1:x2] += r_patch * weight
                weight_map[:, :, y1:y2, x1:x2] += weight

            # Post-processing: normalize by weight map and clamp values
            restored = restored / weight_map.clamp(min=1e-6)
            restored = torch.clamp(restored, 0., 1.)

            # Get metrics and save results
            out_symm = 2 * (torch.clamp(restored, 0, 1) - 0.5)
            gt_symm = 2 * (gt_image - 0.5)
            lpips_avg += loss_fn.forward(out_symm, gt_symm).item()

            ssim_avg += ssim(gt_image.detach(), restored.detach(), data_range=1, size_average=True)

            restored = restored.cpu().detach()
            gt_image = gt_image.cpu().detach()
            in_image = in_image.cpu().detach()

            restored = (np.transpose(np.array(restored)[0], (1, 2, 0)) * 255).astype(np.uint8)
            gt_image = (np.transpose(np.array(gt_image)[0], (1, 2, 0)) * 255).astype(np.uint8)
            in_image = (np.transpose(np.array(in_image)[0], (1, 2, 0)) * 255).astype(np.uint8)

            cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_result.png',
                        cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

            # Save GT and noisy
            if opt.save_gt_noisy:
                cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_gt.png',
                            cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_noisy.png',
                            cv2.cvtColor(in_image, cv2.COLOR_RGB2BGR))

            psnr = get_psnr(restored, gt_image)
            log.info(f'{str(idx).zfill(4)}.png: {psnr:.4f}')
            psnr_avg += psnr

            num_data += 1

            print("Current average psnr : " + str(psnr_avg / num_data))

        psnr_avg /= num_data
        log.info('-' * 40)
        log.info(f'Average PSNR: {psnr_avg:.4f}')

        ssim_avg /= num_data
        log.info('-' * 40)
        log.info(f'Average SSIM: {ssim_avg:.4f}')

        lpips_avg /= num_data
        log.info('-' * 40)
        log.info(f'Average LPIPS: {lpips_avg:.4f}')


if __name__ == '__main__':
    main()
