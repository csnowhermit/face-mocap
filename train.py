from __future__ import print_function
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import json
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
from pytorch3d.io import load_obj

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import lossfunc

from dataset import Face_Dataset
from mobilenetV2 import MobileNetV2
from landmarkLoss import LandmarkLoss
from eyedistanceLoss import EyeDistanceLoss
from lipdistanceLoss import LipDistanceLoss
from photonertrictexLoss import PhotometricTexLoss
from emotionLoss import EmotionLoss

from common import vis_overlay

'''
    CUDA_VISIBEL_DEVICES=0 nohup python -u train_mobilenet.py >> train.log &
'''

os.environ['CUDA_VISIBEL_DEVICES'] = '0'

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 加载模板
def load_BS_template():
    # 标准模板脸
    obj_filename = "./blend51_aligned/head_template.obj"
    verts, faces, aux = load_obj(obj_filename)
    verts = verts.cpu().numpy()
    mean = np.mean(verts, axis=0)
    print("标准模板脸：verts:", verts.shape, ", mean:", mean)

    verts = verts - mean
    return verts

# 加载BS
def load_BS():
    bsList = []

    # # 标准模板脸
    # obj_filename = "./blend51_aligned/head_template.obj"
    # verts, faces, aux = load_obj(obj_filename)
    # verts = verts.cpu().numpy()
    # center = np.mean(verts, axis=0)
    # print("标准模板脸：verts:", verts.shape, ", center:", center)
    #
    # verts = verts - center
    # bsList.append(verts.reshape(5023 * 3))

    bsfile_List = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
              'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
              'eyeLookDownRight',
              'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
              'eyeLookUpRight',
              'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
              'jawOpen',
              'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
              'mouthFrownRight',
              'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
              'mouthPressRight',
              'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
              'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
              'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

    for bs in bsfile_List:
        obj_filename = "./blend51_aligned/%s.obj" % bs
        obj_verts, _, _ = load_obj(obj_filename)
        obj_verts = obj_verts.cpu().numpy()
        mean = np.mean(obj_verts, axis=0)

        # # 先把模板和51个blendshape的原点统一了
        # obj_verts = obj_verts - np.mean(obj_verts, axis=0) + center    # 减去自己的中心，这51个blendshape都挪到同一个地方；再加上模板的中心，表示原点和模板处于同一个点
        # expression = obj_verts - verts - mean    # 我们每个表情应该减去标准脸，拿到的表情才是正交的
        expression = obj_verts - mean
        expression = expression.reshape(5023 * 3)
        bsList.append(expression)

    B = np.array(bsList)    # [51, 5023*3]
    print(B.shape)
    B = B.transpose((1, 0))    # 调换维度顺序，[15069, 51]
    # B = B.reshape(-1, 51)
    print(B.shape)
    return B

# Batch size during training
batch_size = 8
image_size = 224
num_epochs = 500
lr = 1e-4
ngpu = 1
num_classes = 52
continue_training = False    # 是否接着上次的训练
alpha = 1e-2
beta = 1e-3
gama = 1e-3
delta = 1e-3
epsilon = 1e-4
zeta = 1e-6

reg_shape_rate = 1e-4
reg_exp_rate = 1e-4
reg_tex_rate = 1e-4
reg_light_rate = 1e-4



# root_path = "C:/workspace/dataset/bs_dataset_align_20220608/"
root_path = "C:/workspace/dataset/bs_dataset_align_20220530/"
pretrained_model = "./checkpoint/mobilenet_v2-b0353104.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# run DECA
deca = DECA(config=deca_cfg, device='cuda:0')

train_dataset = Face_Dataset(image_root=root_path + "face_train/",
                             mask_root=root_path + "mask/",
                             param_root=root_path + "param.json",
                             landmark_root=root_path + "face_landmark.json",
                             shape_root=root_path + "face_shape.json",
                             pose_root=root_path + "face_pose.json",
                             cam_root=root_path + "face_cam.json",
                             detail_root=root_path + "face_detail.json",
                             light_root=root_path + "face_light.json",
                             tex_root=root_path + "face_tex.json",
                             image_size=image_size, mode="train", device=device)
val_dataset = Face_Dataset(image_root=root_path + "face_val/",
                           mask_root=root_path + "mask/",
                           param_root=root_path + "param.json",
                           landmark_root=root_path + "face_landmark.json",
                           shape_root=root_path + "face_shape.json",
                           pose_root=root_path + "face_pose.json",
                           cam_root=root_path + "face_cam.json",
                           detail_root=root_path + "face_detail.json",
                           light_root=root_path + "face_light.json",
                           tex_root=root_path + "face_tex.json",
                           image_size=image_size, mode="val", device=device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# B_template = load_BS_template()    # [5023, 3]
# B = load_BS()    # [15069, 51]
#
# B_template = torch.from_numpy(B_template)
# B = torch.from_numpy(B)
# B_template = B_template.to(device)
# B = B.to(device)


model = MobileNetV2(num_classes=num_classes)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 10, gamma=0.9)    # 每10 epoch衰减到0.9

# 从预训练模型开始
if len(pretrained_model) > 0:
    if continue_training is False:
        state_dict = torch.load(pretrained_model, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if k not in ['classifier.1.weight', 'classifier.1.bias']}
        state_dict.update(state_dict)
        model.load_state_dict(state_dict, strict=False)  # strict=False，不严格匹配
        del state_dict
    else:
        checkpoint = torch.load(pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        del checkpoint


# def loss_func(pred, label):
#     diff = torch.abs(pred - label) / torch.abs(label)
#     return torch.mean(diff)

def smape(pred, label):
    loss = 2 * torch.abs(pred - label) / (torch.abs(pred) + torch.abs(label))
    return torch.mean(loss)

loss_func = nn.L1Loss()
landmark_loss_func = LandmarkLoss()
eye_loss_func = EyeDistanceLoss()
lip_loss_func = LipDistanceLoss()
phototex_loss_func = PhotometricTexLoss()    # 光学度纹理误差
observeLoss = nn.L1Loss()    # 观测损失
identityLoss = lossfunc.VGGFace2Loss(deca_cfg.model.fr_model_path)    # 身份损失
emoLoss = EmotionLoss(device=device)

# emotion loss要求256*256的
trans = T.Compose([
    T.Resize((256, 256))
])




best_eval_loss = 999999    # 最好的loss，val loss mean只要比该值小，则保存
best_observe_loss = 999999    # 最好的观测损失

total_step = len(train_dataloader)
model.train()
train_loss_list = []
val_loss_list = []
observe_loss_list = []    # 观测损失列表
for epoch in range(num_epochs):
    start = time.time()

    for i, codedict in enumerate(train_dataloader):
        optimizer.zero_grad()
        img = codedict['img'].to(device)
        params = codedict['param'].to(device)
        shape = codedict['shape'].to(device)
        landmark_gt = codedict['cropped_kpt'].to(device)
        pose = codedict['pose'].to(device)
        cam = codedict['cam'].to(device)
        mask = codedict['cropped_mask'].to(device)

        # landmark_gt = landmark_gt.view(-1, landmark_gt.shape[-2], landmark_gt.shape[-1])

        outputs = model(img)    # [bs, 52]
        # 1.bs的损失
        loss1 = loss_func(outputs, params)

        # 计算predict_landmark [bs, 68, 3]
        # outputs = outputs[:, 0:-1]    # 去掉最后一个：tongueOut
        outputs = outputs[:, 0:-1] - 1    # 去掉最后一个：tongueOut，bs范围应该为[0, 1]，而非[1, 2]
        verts, predicted_landmark2d, predicted_landmark3d = deca.flame(shape_params=shape,
                                                                       expression_params=outputs,
                                                                       pose_params=pose)

        # 2.landmark的损失
        landmark_loss = landmark_loss_func(predicted_landmark2d, cam, landmark_gt)

        # 3.eye_distance loss
        eye_distance_loss = eye_loss_func(predicted_landmark2d, landmark_gt)

        # 4.lip_distance loss
        lip_distance_loss = lip_loss_func(predicted_landmark2d, landmark_gt)

        # 5.photometric_texture loss
        opdict = deca.decode_render(codedict, verts, predicted_landmark2d, predicted_landmark3d, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)    # 这里decode和DECA保持一致

        opdict['images'] = img    # 这里img应改为224的
        opdict['lmk'] = landmark_gt

        mask_face_eye = F.grid_sample(deca.uv_face_eye_mask.expand(codedict['img'].shape[0], -1, -1, -1), opdict['grid'].detach(), align_corners=False)    # 这里应为实际读到的batch_size
        predicted_images = opdict['rendered_images'] * mask_face_eye * opdict['alpha_images']
        opdict['predicted_images'] = predicted_images

        mask = mask[:, None, :, :]    # 使用Seg
        phototex_loss = phototex_loss_func(opdict['predicted_images'], opdict['images'], mask)

        # 6.id loss
        shading_images = deca.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
        albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
        overlay = albedo_images * shading_images * mask_face_eye + img * (1 - mask_face_eye)    # overlay [bs, 3, 224, 224]
        id_loss = identityLoss(overlay, img) * deca_cfg.loss.id

        # # 这里对overlay可视化一下，看算出的是什么东西
        # vis_overlay(overlay)
        # exit(0)

        # 7.参考EMOCA，增加EMOTION loss
        emo_loss = emoLoss(trans(overlay), trans(codedict['origin_image']))


        # 8.正则化惩罚项：
        reg_shape_loss = (torch.sum(codedict['shape'] ** 2) / 2) * deca_cfg.loss.reg_shape
        reg_exp_loss = (torch.sum(outputs ** 2) / 2) * deca_cfg.loss.reg_exp    # 这里算exp的reg loss，用的是backbone的输出，用的是前51个，不包括tongout
        reg_tex_loss = (torch.sum(codedict['tex'] ** 2) / 2) * deca_cfg.loss.reg_tex
        reg_light_loss = ((torch.mean(codedict['light'], dim=2)[:, :, None] - codedict['light']) ** 2).mean() * deca_cfg.loss.reg_light

        loss = loss1 + alpha * landmark_loss + beta * eye_distance_loss + gama * lip_distance_loss + delta * phototex_loss + epsilon * id_loss + zeta * emo_loss    # 先加各种损失
        loss_reg = reg_shape_rate * reg_shape_loss + reg_exp_rate * reg_exp_loss + reg_tex_rate * reg_tex_loss + reg_light_rate * reg_light_loss
        loss = loss + loss_reg    # 再加正则损失项

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_list.append(loss.item())

        if (i % 10) == 0:
            print('Epoch [{}/{}], Step [{}/{}], BSLoss: {:.4f}, landmark_loss: {:.4f}, eye_loss: {:.4f}, lip_loss: {:.4f}, phototex_loss: {:.4f}, id_loss: {:.4f}, emo_loss: {:.4f}, reg_loss: {:.4f}, FinalLoss: {:.4f}, spend time: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item(), landmark_loss.item(), eye_distance_loss.item(), lip_distance_loss.item(), phototex_loss.item(), id_loss.item(), emo_loss.item(), loss_reg.item(), loss.item(), time.time() - start))
            start = time.time()

    model.eval()
    with torch.no_grad():
        val_loss = []
        landmarkloss_list = []
        eyeloss_list = []
        liploss_list = []
        phototexloss_list = []
        idloss_list = []
        emoloss_list = []
        regloss_list = []
        observe_list = []    # 观测损失列表
        start = time.time()
        for i, codedict in enumerate(val_dataloader):
            img = codedict['img'].to(device)
            params = codedict['param'].to(device)
            shape = codedict['shape'].to(device)
            landmark_gt = codedict['cropped_kpt'].to(device)
            pose = codedict['pose'].to(device)
            cam = codedict['cam'].to(device)
            mask = codedict['cropped_mask'].to(device)

            outputs = model(img)

            # 这里先算观测损失
            observe_list.append(observeLoss(outputs, params).item())

            # 1.bs的损失
            loss1 = loss_func(outputs, params)

            # 计算predict_landmark [bs, 68, 3]
            # outputs = outputs[:, 0:-1]    # 去掉最后一个：tongueOut
            outputs = outputs[:, 0:-1] - 1  # 去掉最后一个：tongueOut，bs范围应该为[0, 1]，而非[1, 2]
            verts, predicted_landmark2d, predicted_landmark3d = deca.flame(shape_params=shape,
                                                                           expression_params=outputs,
                                                                           pose_params=pose)

            # 2.landmark的损失
            landmark_loss = landmark_loss_func(predicted_landmark2d, cam, landmark_gt)

            # 3.eye_distance loss
            eye_distance_loss = eye_loss_func(predicted_landmark2d, landmark_gt)

            # 4.lip_distance loss
            lip_distance_loss = lip_loss_func(predicted_landmark2d, landmark_gt)

            # 5.photometric_texture loss
            opdict = deca.decode_render(codedict, verts, predicted_landmark2d, predicted_landmark3d, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)  # 这里decode和DECA保持一致

            opdict['images'] = img  # 这里img应改为224的
            opdict['lmk'] = landmark_gt

            mask_face_eye = F.grid_sample(deca.uv_face_eye_mask.expand(codedict['img'].shape[0], -1, -1, -1),
                                          opdict['grid'].detach(), align_corners=False)  # 这里应为实际读到的batch_size
            predicted_images = opdict['rendered_images'] * mask_face_eye * opdict['alpha_images']
            opdict['predicted_images'] = predicted_images

            mask = mask[:, None, :, :]  # 使用Seg
            phototex_loss = phototex_loss_func(opdict['predicted_images'], opdict['images'], mask)

            # 6.id loss
            shading_images = deca.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
            albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
            overlay = albedo_images * shading_images * mask_face_eye + img * (1 - mask_face_eye)
            id_loss = identityLoss(overlay, img) * deca_cfg.loss.id

            # 7.参考EMOCA，增加EMOTION loss
            emo_loss = emoLoss(trans(overlay), trans(codedict['origin_image']))


            # 8.正则化惩罚项：
            reg_shape_loss = (torch.sum(codedict['shape'] ** 2) / 2) * deca_cfg.loss.reg_shape
            reg_exp_loss = (torch.sum(
                outputs ** 2) / 2) * deca_cfg.loss.reg_exp  # 这里算exp的reg loss，用的是backbone的输出，用的是前51个，不包括tongout
            reg_tex_loss = (torch.sum(codedict['tex'] ** 2) / 2) * deca_cfg.loss.reg_tex
            reg_light_loss = ((torch.mean(codedict['light'], dim=2)[:, :, None] - codedict[
                'light']) ** 2).mean() * deca_cfg.loss.reg_light

            loss = loss1 + alpha * landmark_loss + beta * eye_distance_loss + gama * lip_distance_loss + delta * phototex_loss + epsilon * id_loss + zeta * emo_loss  # 先加各种损失
            loss_reg = reg_shape_rate * reg_shape_loss + reg_exp_rate * reg_exp_loss + reg_tex_rate * reg_tex_loss + reg_light_rate * reg_light_loss
            loss = loss + loss_reg  # 再加正则损失项


            val_loss.append(loss.item())
            landmarkloss_list.append(landmark_loss.item())
            eyeloss_list.append(eye_distance_loss.item())
            liploss_list.append(lip_distance_loss.item())
            phototexloss_list.append(phototex_loss.item())
            idloss_list.append(id_loss.item())
            emoloss_list.append(emo_loss.item())
            regloss_list.append(loss_reg.item())

        curr_val_loss = np.mean(val_loss)
        curr_landmark_loss = np.mean(landmarkloss_list)
        curr_eye_loss = np.mean(eyeloss_list)
        curr_lip_loss = np.mean(liploss_list)
        curr_phototex_loss = np.mean(phototexloss_list)
        curr_id_loss = np.mean(idloss_list)
        curr_emo_loss = np.mean(emoloss_list)
        curr_reg_loss = np.mean(regloss_list)
        curr_observe_loss = np.mean(observe_list)
        print('Epoch [{}/{}], val_loss: {:.6f}, landmark_loss: {:.6f}, eye_loss: {:.6f}, lip_loss: {:.6f}, phototex_loss: {:.6f}, id_loss: {:.6f}, emo_loss: {:.6f}, reg_loss: {:.6f}, observe_loss: {:.6f}, time: {:.4f}'
              .format(epoch + 1, num_epochs, curr_val_loss, curr_landmark_loss, curr_eye_loss, curr_lip_loss, curr_phototex_loss, curr_id_loss, curr_emo_loss, curr_reg_loss, curr_observe_loss, time.time() - start))
        val_loss_list.append(curr_val_loss)  # 验证集也保存平均损失
        observe_loss_list.append(curr_observe_loss)    # 观测损失也算平均

        # 这里改为观测损失下降就保存，而不是综合val loss
        if curr_observe_loss < best_observe_loss:    # 只要损失下降就保存
            best_observe_loss = curr_observe_loss    # 保存当前的loss为最好
            torch.save({
                    "curr_epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_eval_loss": best_eval_loss,
                    "lr": scheduler.get_last_lr()
                },root_path + 'checkpoint/mobilenet_{}_loss_{:.6f}_{:.6f}.pt'.format(epoch, curr_val_loss, curr_observe_loss))
        if epoch >= 1:
            plt.figure()
            plt.subplot(131)
            plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
            plt.subplot(132)
            plt.plot(np.arange(0, len(val_loss_list)), val_loss_list)
            plt.subplot(133)
            plt.plot(np.arange(0, len(observe_loss_list)), observe_loss_list)
            plt.savefig(root_path + "metrics.png")
            plt.close("all")
    model.train()
