from einops.einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch

from einops import rearrange


class CropCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_pred, pos_gt):
        mask_pred_temp = rearrange(mask_pred, 'b r h w -> (b r) h w')
        pos_gt_temp = rearrange(pos_gt, 'b r n -> (b r) n')
        gt_mask = torch.zeros_like(mask_pred_temp)

        for idx, individual_gt_mask in enumerate(gt_mask):
            individual_gt_pos = pos_gt_temp[idx]
            individual_gt_mask[individual_gt_pos[0]:individual_gt_pos[2] +
                               1, individual_gt_pos[1]:individual_gt_pos[3]+1] = 1

        return -(torch.log(mask_pred_temp)*gt_mask +
                 torch.log((1 - mask_pred_temp))*(1-gt_mask)).mean()
