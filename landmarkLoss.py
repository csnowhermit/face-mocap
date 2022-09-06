import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import lossfunc

class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.loss_lmk = 1.0

    def forward(self, predicted_landmark2d, cam, landmark_gt):

        # projected
        predicted_landmark2d = util.batch_orth_proj(predicted_landmark2d, cam)[:, :, :2];predicted_landmark2d[:, :, 1:] = -predicted_landmark2d[:, :, 1:]  # ; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2    # projection后才是[1, 68, 2]

        # 2.landmark2d和landmark_gt的loss
        landmark_loss = lossfunc.weighted_landmark_loss(predicted_landmark2d, landmark_gt) * self.loss_lmk

        return landmark_loss

if __name__ == '__main__':
    shape = torch.randn([8, 100], dtype=torch.float32)
    shape = shape.cuda()

    exp = torch.randn([8, 51], dtype=torch.float32)
    exp = exp.cuda()

    pose = torch.randn([8, 6], dtype=torch.float32)
    pose = pose.cuda()

    cam = torch.randn([8, 3], dtype=torch.float32)
    cam = cam.cuda()

    landmark_gt = torch.randn([8, 68, 2], dtype=torch.float32)
    landmark_gt = landmark_gt.cuda()

    loss = LandmarkLoss(device='cuda:0')
    landmark_loss = loss(shape, exp, pose, cam, landmark_gt)
    print("landmark_loss:", landmark_loss)


