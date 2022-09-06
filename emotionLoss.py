import torch
import torch.nn as nn

from emonet.models import EmoNet

class EmotionLoss(nn.Module):
    def __init__(self, device):
        super(EmotionLoss, self).__init__()
        self.device = device

        self.model = EmoNet(n_expression=8).to(device)
        self.model.load_state_dict(torch.load("./checkpoint/emonet_8.pth"))
        self.model.eval()

        self.loss_func = nn.L1Loss()

    '''
        这里predict_image为DECA中overlay
    '''
    def forward(self, predict_image, img):
        pre_emo = self.model(predict_image)['expression']
        img_emo = self.model(img)['expression']
        return self.loss_func(pre_emo, img_emo)
