import os
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.utils as vutils

# 调色板
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(config.num_classes)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")

'''
    可视化
    :param overlay [bs, 3, 224, 224]
'''
def vis_overlay(overlay):
    show_img_list = [overlay[i] * 255. for i in range(overlay.shape[0])]

    label_show = vutils.make_grid(show_img_list, nrow=4, padding=2, normalize=True).cpu()    # nrow代表每行几个

    vutils.save_image(label_show, "./vis_overlay.png")


