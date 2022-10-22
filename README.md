# face-mocap

​	人脸单目动捕算法。

# 1、模型

​	backbobe：mobilenetV2

​	输入：RGB图像

​	输出：52个blendshape

# 2、损失函数

主要损失：

​	blendshape回归损失：L1Loss；

辅助损失：

​	landmark损失；

​	双眼间距损失；

​	嘴角间距损失；

​	图像纹理损失；

​	身份id损失；

​	人脸表情损失；

# 3、指标
blendshape回归损失：0.009663

# 4、纯数学方式解算blendshape

``` python
python blendshape_matrix_solve.py
```

