import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import glob

 # 程序流程
 # 1.准备好一系列来相机标定的图片
 # 2.对每张图片提取角点信息
 # 3.由于角点信息不够精确，进一步提取亚像素角点信息
 # 4.在图片中画出提取出的角点
 # 5.相机标定
 # 6.对标定结果评价，计算误差
 # 7.使用标定结果对原图片进行校正
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
path = 'D:/files/cuhk/cv/assignment/cv_coding/source'   # 文件路径
objp = np.zeros((8 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:7].T.reshape(-1, 2)
# mgrid是meshgrid的缩写，生成的是坐标网格，输出的参数是坐标范围，得到的网格的点坐标
op = [] # 存储世界坐标系的坐标X，Y，Z，在张正友相机标定中Z=0
imgpoints = []  # 像素坐标系中角点的坐标
for i in os.listdir(path+"/image"):
    #读取每一张图片
    file = '/'.join((path, "image/{}".format(i)))
    img = cv2.imread(file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 确定输入图像中是否有棋盘格图案，并检测棋盘格的内角点
    ret, corners = cv2.findChessboardCorners(gray, (8, 7), None)
    if ret == True: # 如果所有的内角点都找到了
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)   # 提取亚像素角点信息
        imgpoints.append(corners2)
        op.append(objp)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8, 7), corners2, ret)
        cv2.imwrite('/'.join((path, '/result/r{}'.format(i))),img)
        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)
        cv2.waitKey(50)



cv2.destroyAllWindows()

# 相机标定的核心
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(op, imgpoints, gray.shape[::-1], None, None)
# mtx是内参矩阵
# dist为相机的畸变参数矩阵
# rvecs为旋转向量
# tvecs为位移向量
print(mtx)
tot_error = 0
for i in range(len(op)):
    imgpoints2, _ = cv2.projectPoints(op[i], rvecs[i], tvecs[i], mtx, dist) #对空间中的三维坐标点进行反向投影
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)   # 平均平方误差（重投影误差）
    tot_error += error
print((tot_error / len(op))**0.5)
