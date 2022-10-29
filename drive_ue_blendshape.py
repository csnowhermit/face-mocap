import os
import socket
import time
import random
import numpy as np
from PIL import ImageGrab

from common.pylivelinkface import PyLiveLinkFace, FaceBlendShape

'''
    使用livelink，通过blendshape驱动ue中的虚拟人
    ue端使用自带的livelink即可
'''

if __name__ == '__main__':
    py_face = PyLiveLinkFace()

    UDP_IP = "192.168.109.1"    # 这里IP写本机的局域网IP，不能写127.0.0.1
    UDP_PORT = 11111    # ue端自带的livelink的端口

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((UDP_IP, UDP_PORT))

    while True:
        for idx, item in enumerate(FaceBlendShape):
            py_face.set_blendshape(item, random.uniform(-1, 1))
        s.sendall(py_face.encode())
        print("已发送：", py_face.encode())
        time.sleep(0.05)





    # for file in os.listdir(base_path):
    #     read_json = read_yuhaiyang(os.path.join(base_path, file))
    #
    #     for i, item in enumerate(FaceBlendShape):
    #         # print(item.name, read_json[item.name])
    #         if item.name in model_bsList:
    #             py_face.set_blendshape(item, read_json[item.name])
    #     s.sendall(py_face.encode())
    #     print("已发送：", file)
    #     time.sleep(0.05)


