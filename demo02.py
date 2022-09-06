import os
import json
import torch
from pytorch3d.io import load_obj, save_obj

# verts, faces, _ = load_obj("./blend51_aligned/head_template.obj")
# print(verts.shape)
# print(faces[0].shape)
#
# jaw_verts, faces, _ = load_obj("./blend51_aligned/jawOpen.obj")
#
# jaw_verts = jaw_verts - verts
#
# save_obj("./zengliang_jawOpen.obj", jaw_verts, faces[0])

x = torch.ones([2, 3], dtype=torch.uint8)
print(x)

i = torch.tensor(1, dtype=torch.uint8)

if x.all() == i:    # 所有的都是1
    print("==1")

if x.any() == i:    # 存在有1
    print("any 1")


y = x * 255.
print(y.shape)
print(y)
