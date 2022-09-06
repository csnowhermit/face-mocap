import numpy as np
import torch

# lmk = np.load("D:/n000002#0001_01.npy")[0][:, :2]
# print(lmk.shape)
# # print(lmk.shape[-2], lmk.shape[-1])
# print(lmk)
# # print("====================")
# # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
# # print(lmk)

x = torch.randn([3, 4, 2], dtype=torch.float32)
print(x)
print("=========================")

y = x.view(3, -1)
print(y)
print("=========================")

# print(y.T)
# print("=========================")

z = y.reshape(-1, 4, 2)
print(z)
print("=========================")

B0 = torch.tensor([[1, 2],
                   [3, 4],
                   [5, 6],
                   [7, 8]], dtype=torch.float32)

z = z + B0
print(z)
print(z.shape)