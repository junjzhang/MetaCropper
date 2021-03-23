import torch.nn.functional as F
import torch.nn as nn
import torch


class CropCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_pred, pos_gt):
        gt_mask = torch.zeros(mask_pred)
        for idx, individual_gt_mask in enumerate(gt_mask):
            individual_gt_pos = pos_gt[idx]
            individual_gt_mask[individual_gt_pos[0]:individual_gt_pos[2] +
                               1, individual_gt_pos[1]:individual_gt_pos[3]+1] = 1
        return torch.log(mask_pred)*gt_mask + torch.log((1 - mask_pred))*(1-gt_mask).mean()
