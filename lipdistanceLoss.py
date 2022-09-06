import torch
import torch.nn as nn

class LipDistanceLoss(nn.Module):
    def __init__(self, device='cuda:0'):
        super(LipDistanceLoss, self).__init__()
        self.lipd_loss = 0.5
        self.device = device
        self.loss_func = nn.L1Loss()

    def forward(self, predict_landmark, landmark_gt):
        pred_lipd = lip_dis(predict_landmark[:, :, :2])
        gt_lipd = lip_dis(landmark_gt[:, :, :2])
        return self.loss_func(pred_lipd, gt_lipd) * self.lipd_loss


def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    lip_up = landmarks[:,[61, 62, 63], :]
    lip_down = landmarks[:,[67, 66, 65], :]
    dis = torch.sqrt(((lip_up - lip_down)**2).sum(2)) #[bz, 4]
    return dis

