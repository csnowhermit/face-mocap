import torch
import torch.nn as nn

'''
    光学度纹理误差
'''

class PhotometricTexLoss(nn.Module):
    def __init__(self):
        super(PhotometricTexLoss, self).__init__()
        self.photo_loss = 2.0
        self.loss_func = nn.L1Loss()


    def forward(self, predict_images, images, mask):
        # return mask * (predict_images - images).abs().mean() * self.photo_loss    # 渲染出的图，相对于原图的误差
        return (mask * (predict_images - images).abs()).mean() * self.photo_loss
        # return mask * self.loss_func(predict_images, images) * self.photo_loss

