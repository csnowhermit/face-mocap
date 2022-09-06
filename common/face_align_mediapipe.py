import os
import cv2
# import dlib
import numpy
import sys
import matplotlib.pyplot as plt

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

'''
    人脸对齐功能
'''

PREDICTOR_PATH = r"./checkpoint/shape_predictor_68_face_landmarks.dat"  # 68个关键点landmarks的模型文件
SCALE_FACTOR = 1 # 图像的放缩比
FEATHER_AMOUNT = 15  # 羽化边界范围，越大，羽化能力越大，一定要奇数，不能偶数

template_w = 410
template_h = 512


# #　68个点
# FACE_POINTS = list(range(17, 68))  # 脸
# MOUTH_POINTS = list(range(48, 61))  # 嘴巴
# RIGHT_BROW_POINTS = list(range(17, 22))  # 右眉毛
# LEFT_BROW_POINTS = list(range(22, 27))  # 左眉毛
# RIGHT_EYE_POINTS = list(range(36, 42))  # 右眼睛
# LEFT_EYE_POINTS = list(range(42, 48))  # 左眼睛
# NOSE_POINTS = list(range(27, 35))  # 鼻子
# JAW_POINTS = list(range(0, 17))  # 下巴
#
# # 所有关键点都要对齐
# ALIGN_POINTS = (FACE_POINTS + LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
#                 RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS)
#
# # 定义用于颜色校正的模糊量，作为瞳孔距离的系数
# COLOUR_CORRECT_BLUR_FRAC = 0.6
#
# # 实例化脸部检测器
# detector = dlib.get_frontal_face_detector()
# # 加载训练模型
# # 并实例化特征提取器
# predictor = dlib.shape_predictor(PREDICTOR_PATH)

############ mediapipe ########## 75个关键点
# face_counter = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
#                 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
#                 300, 293, 334, 296, 336,
#                 70, 63, 105, 66, 107,
#                 6, 197, 195, 5, 4, 1, 20, 250,
#                 33, 161, 159, 157, 133, 154, 145, 163,
#                 362, 384, 386, 388, 263, 390, 374, 381]
# lips_upper = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
# left_brow_2 = [300, 293, 334, 296, 336]
# right_brow_2 = [70, 63, 105, 66, 107]
# nose_counter = [6, 197, 195, 5, 4, 1, 20, 250]
# left_eye = [33, 161, 159, 157, 133, 154, 145, 163]
# right_eye = [362, 384, 386, 388, 263, 390, 374, 381]
#
#
# ALIGN_POINTS = (face_counter + lips_upper + left_brow_2 + right_brow_2 + left_eye + right_eye + nose_counter)
left_eye = [33, 133]
right_eye = [263, 362]
nose = [1]
mouth = [61, 291]

ALIGN_POINTS = (left_eye + right_eye + nose + mouth)

holistic = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True)


# 定义了两个类处理意外
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass

def get_landmarks(im):
    '''
    通过predictor 拿到75 landmarks
    '''
    results = holistic.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # if len(results) > 1:
    #     raise TooManyFaces
    # if len(results) == 0:
    #     raise NoFaces

    landmark_result = numpy.zeros((len(ALIGN_POINTS) - 2, 2))
    h, w, c = im.shape
    # print(im.shape)
    index = 0
    # print(landmark_result.shape)

    landmark_result[0][0] = (results.face_landmarks.landmark[left_eye[0]].x + results.face_landmarks.landmark[left_eye[1]].x) * w / 2
    landmark_result[0][1] = (results.face_landmarks.landmark[left_eye[0]].y + results.face_landmarks.landmark[left_eye[1]].y) * h / 2

    landmark_result[1][0] = (results.face_landmarks.landmark[right_eye[0]].x + results.face_landmarks.landmark[right_eye[1]].x) * w / 2
    landmark_result[1][1] = (results.face_landmarks.landmark[right_eye[0]].y + results.face_landmarks.landmark[right_eye[1]].y) * h / 2

    landmark_result[2][0] = results.face_landmarks.landmark[nose[0]].x * w
    landmark_result[2][1] = results.face_landmarks.landmark[nose[0]].y * h

    landmark_result[3][0] = results.face_landmarks.landmark[mouth[0]].x * w
    landmark_result[3][1] = results.face_landmarks.landmark[mouth[0]].y * h

    landmark_result[4][0] = results.face_landmarks.landmark[mouth[1]].x * w
    landmark_result[4][1] = results.face_landmarks.landmark[mouth[1]].y * h

    return numpy.matrix(landmark_result)

    # rects = detector(im, 1)
    #
    # if len(rects) > 1:
    #     raise TooManyFaces
    # if len(rects) == 0:
    #     raise NoFaces

    # return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])  # 75*2的矩阵

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)    # [468, 2]

    return im, s

# 返回一个仿射变换
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)    # 人脸的指定关键点 [75, 2]
    points2 = points2.astype(numpy.float64)    # [75, 2]

    # 每张脸各自做各自的标准化
    c1 = numpy.mean(points1, axis=0)    # 分别算x和y的均值
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1    # 浮动于均值的部分,[43, 2]
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1    #
    points2 /= s2

    # print("points1.T:", points1.T.shape)
    # print("points2:", points2.shape)
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    # U, S, Vt = numpy.linalg.svd(tmp)

    R = (U * Vt).T    # [2, 2]

    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])

'''
    对齐函数
    由 get_face_mask 获得的图像掩码还不能直接使用，因为一般来讲用户提供的两张图像的分辨率大小很可能不一样，而且即便分辨率一样，
    图像中的人脸由于拍摄角度和距离等原因也会呈现出不同的大小以及角度，所以如果不能只是简单地把第二个人的面部特征抠下来直接放在第一个人脸上，
    我们还需要根据两者计算所得的面部特征区域进行匹配变换，使得二者的面部特征尽可能重合。

    仿射函数，warpAffine，能对图像进行几何变换
        三个主要参数，第一个输入图像，第二个变换矩阵 np.float32 类型，第三个变换之后图像的宽高
'''
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)    # [512, 512, 3]
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# 人脸对齐函数
def face_Align(Base_path, cover_path):
    # im1, landmarks1 = read_im_and_landmarks(Base_path)  # 模板图 [1024, 1024, 3], [75, 2]
    im2, landmarks2 = read_im_and_landmarks(cover_path)  # 要对齐的图 [1440, 1080, 3], [75, 2]

    landmarks1 = [(0.31556875000000000, 0.4615741071428571),
                  (0.68262291666666670, 0.4615741071428571),
                  (0.50026249999999990, 0.6405053571428571),
                  (0.34947187500000004, 0.8246919642857142),
                  (0.65343645833333330, 0.8246919642857142)]

    landmarks1 = numpy.asmatrix([(x * template_w, y * template_h) for (x, y) in landmarks1])
    im1_shape = (template_h, template_w, 3)

    # 得到仿射变换矩阵
    M = transformation_from_points(landmarks1,
                                   landmarks2)
    warped_im2 = warp_im(im2, M, im1_shape)
    return warped_im2


if __name__ == '__main__':
    # cover_path = 'C:/workspace/dataset/bs_dataset/face_val/Frame5534.jpg'    # 要对齐的人脸
    template_path = './template.jpg'  # 模板
    # warped_mask = face_Align(template_path, cover_path)
    # cv2.imwrite("./result.jpg", warped_mask)
    import time
    base_path = "/dataset/face/"
    save_path = "/dataset/face_align/"
    cnt = 0
    start = time.time()
    for file in os.listdir(base_path):
        cnt += 1
        if cnt % 1 == 0:
            print("Processing...", cnt, "spend time:", time.time() - start)
            start = time.time()
        try:
            warped_mask = face_Align(template_path, os.path.join(base_path, file))
            cv2.imwrite(os.path.join(save_path, file), warped_mask)
        except:
            print(file, " has many face or no face")

