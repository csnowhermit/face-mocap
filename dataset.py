import os
import json
from PIL import Image
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class Face_Dataset(Dataset):
    def __init__(self, image_root, mask_root, param_root, landmark_root, shape_root, pose_root, cam_root, detail_root, light_root, tex_root, image_size=299, mode='train', device=torch.device('cpu')):
        self.image_root = image_root
        self.mask_root = mask_root    # 人脸mask的目录
        self.mode = mode
        self.device = device
        with open(param_root, 'r', encoding='utf-8') as f:
            self.params = json.load(f)

        with open(shape_root, 'r', encoding='utf-8') as f:
            self.shape = json.load(f)

        with open(landmark_root, 'r', encoding='utf-8') as f:
            self.landmark = json.load(f)

        with open(pose_root, 'r', encoding='utf-8') as f:
            self.pose = json.load(f)

        with open(cam_root, 'r', encoding='utf-8') as f:
            self.cam = json.load(f)

        with open(detail_root, 'r', encoding='utf-8') as f:
            self.detail = json.load(f)

        with open(light_root, 'r', encoding='utf-8') as f:
            self.light = json.load(f)

        with open(tex_root, 'r', encoding='utf-8') as f:
            self.tex = json.load(f)

        self.scale = [1.4, 1.8]  # [scale_min, scale_max]
        self.trans_scale = 0.  # [dx, dy]
        self.image_size = image_size
        # 获取文件列表
        self.imglist = []    # 这里只写文件名，不写路径
        for file in os.listdir(self.image_root):
            self.imglist.append(file)

        self.transform = T.Compose([
            T.ToTensor(),
            # T.Grayscale(num_output_channels=3),     # 3通道的灰度图
            # T.RandomRotation(10),    # [-10, 10]度的随机偏转
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),    # 亮度、对比度、饱和度、色调
            # # T.RandomAffine(),    # 暂时用随即偏转代替
            T.Resize((image_size, image_size), interpolation=0),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.trans = T.Compose([
            T.ToTensor()
        ])


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_root, self.imglist[index])).convert('RGB')
        param = torch.tensor([i+1 for i in self.params[self.imglist[index]]])    # 用文件名拿到标签，统一预测[1, 2]区间的值；

        landmark_gt = self.landmark[self.imglist[index]]  # [68, 2]
        landmark_gt = np.array(landmark_gt)

        # 参照DECA，还需按如下方法做norm
        ### crop information
        image = imread(os.path.join(self.image_root, self.imglist[index])) / 255.
        tform = self.crop(image, landmark_gt)
        ## crop
        cropped_kpt = np.dot(tform.params, np.hstack(
            [landmark_gt, np.ones([landmark_gt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
        cropped_kpt = torch.tensor(cropped_kpt)    # [68, 3]


        # 读取mask文件
        vis_parsing_anno = np.load(os.path.join(self.mask_root, self.imglist[index].replace(".jpg", ".npy")))
        mask = np.zeros_like(vis_parsing_anno)    # 创建一个和vis_parsing_anno相同大小、均为0的mask
        mask[vis_parsing_anno>0] = 1    # 脸部位为1，其他部位为0.
        cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_mask = torch.from_numpy(cropped_mask)

        # 做成dict返回
        codedict = {}
        codedict['origin_image'] = self.trans(img).to(self.device)    # 原图用于算emo loss
        codedict['img'] = self.transform(img).to(self.device)
        codedict['param'] = param.to(self.device)
        codedict['shape'] = torch.tensor(self.shape[self.imglist[index]]).to(self.device)
        codedict['cropped_kpt'] = cropped_kpt.to(self.device)
        codedict['pose'] = torch.tensor(self.pose[self.imglist[index]]).to(self.device)
        codedict['cam'] = torch.tensor(self.cam[self.imglist[index]]).to(self.device)
        codedict['cropped_mask'] = cropped_mask.to(self.device)
        codedict['detail'] = torch.tensor(self.detail[self.imglist[index]]).to(self.device)
        codedict['light'] = torch.from_numpy(np.array(self.light[self.imglist[index]], dtype=np.float32)).to(self.device)    # [bs, 9, 3]，这里应该为np.float32
        codedict['tex'] = torch.tensor(self.tex[self.imglist[index]]).to(self.device)

        # return self.transform(img), param, shape, cropped_kpt, pose, cam, cropped_mask
        return codedict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def __len__(self):
        return len(self.imglist)


if __name__ == '__main__':
    root_path = "C:/workspace/dataset/bs_dataset_align_20220530/"
    image_size = 299

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
                                 image_size=image_size, mode="train")
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
                               image_size=image_size, mode="val")

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(len(train_dataloader))
    for i, codedict in enumerate(train_dataloader):
        print(i, codedict['img'].shape, codedict['param'].shape, codedict['shape'].shape, codedict['cropped_kpt'].shape, codedict['pose'].shape, codedict['cam'].shape, codedict['detail'].shape, codedict['light'].shape, codedict['tex'].shape)

