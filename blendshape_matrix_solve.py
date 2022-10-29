from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
import numpy as np
import scipy
from scipy.optimize import nnls
from numpy.linalg import inv    # 求矩阵的逆
import os
import gc
import json
import time
import torch
import torch.nn as nn
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
from skimage.io import imread
from skimage.transform import rescale, estimate_transform, warp
from torch.utils.data import Dataset

# from gdl.datasets.FaceVideoDataModule import add_pretrained_deca_to_path
from gdl.datasets.ImageDatasetHelpers import bbox2point
from gdl.utils.FaceDetector import FAN

import pytorch3d
from pytorch3d.io import load_obj

'''
    纯数学方式求解blendshape
    1.先用emoca得到f；
    2.公式Bw=f，求解w；
'''

'''
    加载bs文件
'''
def load_BS():
    bsList = []

    # 标准模板脸
    obj_filename = "./blend51_aligned/head_template.obj"
    verts, faces, aux = load_obj(obj_filename)
    verts = verts.cpu().numpy()
    center = np.mean(verts, axis=0)
    print("标准模板脸：verts:", verts.shape, ", center:", center)

    verts = verts - center
    bsList.append(verts.reshape(5023 * 3))

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

        # # 做norm
        # expression = (expression - np.mean(expression, axis=0)) / np.std(expression, axis=0)
        #
        # # check
        # print("check %s norm:" % bs)
        # print(np.mean(expression, axis=0))
        # print(np.std(expression, axis=0))

        expression = expression.reshape(5023 * 3)

        # print(bs, expression.shape)
        bsList.append(expression)

    B = np.array(bsList)    # [51, 5023*3]
    print(B.shape)
    B = B.transpose((1, 0))    # 调换维度顺序，[15069, 51]
    # B = B.reshape(-1, 51)
    print(B.shape)
    return B

'''
    准备数据
'''
def prepare_data(input_folder, file):
    max_detection = 20
    scaling_factor = 1.0
    crop_size = 224
    scale = 1.25
    iscrop = True
    resolution_inp = 224
    face_detector = FAN()

    imagepath = os.path.join(input_folder, file)
    imagename = imagepath.split('/')[-1].split('.')[0]

    image = np.array(imread(imagepath))
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(1, 1, 3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    if scaling_factor != 1.:
        image = rescale(image, (scaling_factor, scaling_factor, 1)) * 255.

    h, w, _ = image.shape
    if iscrop:
        # provide kpt as txt file, or mat file (for AFLW2000)
        kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
        kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(kpt_matpath):
            kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        elif os.path.exists(kpt_txtpath):
            kpt = np.loadtxt(kpt_txtpath)
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        else:
            # bbox, bbox_type, landmarks = self.face_detector.run(image)
            bbox, bbox_type = face_detector.run(image)
            if len(bbox) < 1:
                print('no face detected! run original image')
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
                old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            else:
                if max_detection is None:
                    bbox = bbox[0]
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                else:
                    old_size, center = [], []
                    num_det = min(max_detection, len(bbox))
                    for bbi in range(num_det):
                        bb = bbox[0]
                        left = bb[0]
                        right = bb[2]
                        top = bb[1]
                        bottom = bb[3]
                        osz, c = bbox2point(left, right, top, bottom, type=bbox_type)
                    old_size += [osz]
                    center += [c]

        if isinstance(old_size, list):
            size = []
            src_pts = []
            for i in range(len(old_size)):
                size += [int(old_size[i] * scale)]
                src_pts += [np.array(
                    [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2],
                     [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                     [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
        else:
            size = int(old_size * scale)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])
    else:
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

    image = image / 255.
    if not isinstance(src_pts, list):
        DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
        image = None
        torch.cuda.empty_cache()
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                 'image_name': imagename,
                 'image_path': imagepath,
                 # 'tform': tform,
                 # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                 }
    else:
        DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
        dst_images = []
        for i in range(len(src_pts)):
            tform = estimate_transform('similarity', src_pts[i], DST_PTS)
            dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
            dst_image = dst_image.transpose(2, 0, 1)
            dst_images += [dst_image]
        image = None
        torch.cuda.empty_cache()
        dst_images = np.stack(dst_images, axis=0)

        imagenames = [imagename + f"{j:02d}" for j in range(dst_images.shape[0])]
        imagepaths = [imagepath] * dst_images.shape[0]
        return {'image': torch.tensor(dst_images).float(),
                 'image_name': imagenames,
                 'image_path': imagepaths,
                 # 'tform': tform,
                 # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                 }

def main():
    path_to_models = "./assets/EMOCA/models"
    input_folder = 'D:/testdata/dat/'
    model_name = 'EMOCA'

    mode = 'detail'
    # mode = 'coarse'

    # 1.Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    # 2.加载bs矩阵
    B = load_BS()

    template_verts, faces, _ = load_obj("./blend51_aligned/head_template.obj")    # 用模板的脸
    template_verts = template_verts.cpu().numpy()
    template_verts_copy = template_verts.copy()
    template_verts = template_verts.reshape(-1, 1)

    # 3.Run the model on the data
    for file in os.listdir(input_folder):
        # 4.1、读取数据集
        file = "qidewang_Frame6240.jpg"
        batch = prepare_data(input_folder, file)

        vals, visdict = test(emoca, batch['image'])

        f = vals['verts'].detach().cpu().numpy()[0]    # 用flame参数化后的mesh [5023, 3]
        # f_mean = np.mean(f, axis=0)
        # f_std = np.std(f, axis=0)
        # f = (f-f_mean) / f_std
        #
        # # check
        # print("check f norm:")
        # print(np.mean(f, axis=0))
        # print(np.std(f, axis=0))

        f = f.reshape(15069)    # [15069, 1]

        # 公式：Bw = f， 可得w = （B^T * B）^-1 * B^T *f
        # w = np.dot(np.dot(inv(np.dot(B.T, B)), B.T), f)    # [51, 1]
        # w = np.matmul(np.matmul(inv(np.matmul(B.T, B)), B.T), f)  # [51, 1] B*w=f

        # 用最小二乘的办法做
        # w = np.linalg.lstsq(B, f-template_verts, rcond=None)[0]    # 这种方法不保证计算结果是非负的
        w = nnls(B, f)[0]    # non negative least squares

        print("纯数学解算结果：", w)

        L2 = np.mean(np.power((np.matmul(B, w) - f), 2))    # ||Bw - f||^2
        loss = np.matmul(B, w) - f
        print("loss:", loss)

        print("emoca 输出 f 的统计：")
        print("flame f verts:")
        print("mean:", np.mean(f))
        print("std:", np.std(f))
        print("max:", np.max(f))
        print("min:", np.min(f))
        print("==================================")

        tmp = np.matmul(B, w)    # 乘回去，用于评估纯数学方式的损失
        print("纯数学解算的统计信息：")
        print("flame f np.matmul(B, w):")
        print("mean:", np.mean(tmp))
        print("std:", np.std(tmp))
        print("max:", np.max(tmp))
        print("min:", np.min(tmp))
        print("==================================")

        # 分别保存emoca的结果和数学解算后相乘的结果
        pytorch3d.io.save_obj("./f.obj", vals['verts'][0], faces[0])
        tmp = tmp.reshape(5023, 3)
        tmp = torch.tensor(tmp + template_verts_copy, dtype=torch.float32).cuda()
        pytorch3d.io.save_obj("./bw.obj", tmp, faces[0])

        break
    print("Done")


if __name__ == '__main__':
    main()