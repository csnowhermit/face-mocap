import torch
import torch.nn as nn

class EyeDistanceLoss(nn.Module):
    def __init__(self, device='cuda:0'):
        super(EyeDistanceLoss, self).__init__()
        self.eyed_loss = 1.0
        self.device = device
        self.loss_func = nn.L1Loss()

    def forward(self, predict_landmark, landmark_gt):
        pred_eyed = eye_dis(predict_landmark[:, :, :2])
        gt_eyed = eye_dis(landmark_gt[:, :, :2])
        return self.loss_func(pred_eyed, gt_eyed) * self.eyed_loss




def eye_dis(landmarks):
    # left eye:  [38,42], [39,41] - 1
    # right eye: [44,48], [45,47] -1
    eye_up = landmarks[:,[37, 38, 43, 44], :]
    eye_bottom = landmarks[:,[41, 40, 47, 46], :]
    dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
    return dis
