# -*- coding :  utf-8 -*-
# @Data      :  2019-08-16
# @Author    :  zs
# @Email     :  shuaizhang_16@163.com
# @File      :  image_process.py
# Desctiption:  求取图像中物体的边界矩形

import numpy as np
import cv2
import os


def calculatBoundImage(src_Image):
    """
    求取图像中物体的边界矩形框
    :param src_Image: 输出的源图像
    :return: 返回图像中的物体边界矩形
    """

    tmp_image = src_Image.copy()
    # print(tmp_image)
    if (len(tmp_image.shape) == 3):
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
        
    # 自适应阈值进行二值化
    thresh_image = cv2.adaptiveThreshold(tmp_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 10)
    # 形态学操作，闭操作
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    
    # 寻找最外层轮廓
    contours_ls, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # 将各个轮廓的数量存储在一个数组中，目的是找到最大轮廓是属于那个索引
    pnt_cnt_ls = np.array([tmp_contour.shape[0] for tmp_contour in contours_ls])

    # 在原图上进行绘制轮廓
    contour_image = src_Image.copy()
    
    # 找到 最大轮廓所对应的索引 
    contours_idx = np.argmax(pnt_cnt_ls)
    # 绘制最大的轮廓
    contour_image = cv2.drawContours(contour_image, contours_ls, contours_idx, (0, 0, 255))
    
    longest_contour = contours_ls[contours_idx]

    # 在空白图上（全黑图像）以填充模式绘制最大轮廓
    countour_image_gray = np.zeros(src_Image.shape, dtype=np.uint8)
    countour_image_gray = cv2.drawContours(countour_image_gray, contours_ls, contours_idx, (1, 1, 1), cv2.FILLED)
    
    # 把绘制的填充图像放到原图像上，这里是填充部分与原图融合
    obj_image = src_Image * countour_image_gray
    # 返回最长轮廓的外接矩形框数据，bound_box包含x,y,w,h,即左上角坐标，和宽，高
    bound_box = cv2.boundingRect(longest_contour)
    return bound_box, contour_image, obj_image


def rotateImage(src_Image, angle_deg, rotate_center=None):
    """
    对目标图像进行旋转
    :param src_Image: 输入的源图像
    :param angle_deg: 旋转的角度
    :param rotate_center: 旋转的中心
    :return: 旋转后的图片
    """
    (h, w) = src_Image.shape[:2]
    if rotate_center is None:
        rotate_center = ((w -1) / 2, (h - 1) / 2)
     
    # 获得旋转矩阵，(h, w)表示得到和原图大小一样的图像
    rot_mat = cv2.getRotationMatrix2D(rotate_center, angle_deg, 1.0)
    # 获得旋转后的矩阵图像
    rot_iamge = cv2.warpAffine(src_Image, rot_mat, (w, h))
    return rot_iamge


def VideotoImage(video_file, folder_path):
    """
    数据的视频保存为提取之后的物体图
    :param video_file: 视频文件
    :param folder_path: 保存图片的路径
    :return: 保存的图片
    """
    #　获取视频
    video_cap = cv2.VideoCapture(video_file)
    # 设置拆取图像保存名的初始索引
    image_idx = 2000
    while True:
        ret, frame = video_cap.read()
        if (frame is None):
            continue
        bound_box, contour_image, obj_image = calculatBoundImage(frame)
        bound_thres = 4500

        # 如果矩形框的宽或高超过给定的阈值则跳过继续下一个判断
        if (bound_box[2] > bound_thres or bound_box[3] > bound_thres):
            continue
            
        # 绘制矩形框
        contour_image = cv2.rectangle(contour_image, (bound_box[0], bound_box[1]),(bound_box[0] + bound_box[2],bound_box[1] + bound_box[3]), (225, 0, 0), thickness=2)
        #cv2.imshow('frame', contour_image)
        
        # 将图片索引宽度扩展为6位，例原2000,其左边填充两个0，变为002000
        image_name = str(image_idx).zfill(6) + '.jpg'
        image_idx += 1
        # 跳帧读取并保存数据
        if image_idx % 2 == 0:
            cv2.imwrite(folder_path + image_name, obj_image)
        cv2.waitKey(25)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    video_cap.release()


def BatchImageProcess(image_path, folder_path):
    """
    批量图片物体提取，背景为黑色
    :param Image_path: 图片的路径
    :param folder_path: 图像处理之后的保存路径
    :return: 保存的图片
    """
    image_file_list = os.listdir(image_path)
    # 获取物体图像的文件名，即文件的绝对路径，方便读取图像
    image_idx = 0
    for image_name in range(len(image_file_list)):
        obj_image_path = image_path + image_file_list[image_idx]
        src_Image = cv2.imread(obj_image_path)

        bound_box, contour_image, obj_image = calculatBoundImage(src_Image)
        bound_thres = 4500

        if (bound_box[2] > bound_thres or bound_box[3] > bound_thres):
            continue
        contour_image = cv2.rectangle(contour_image, (bound_box[0], bound_box[1]), (bound_box[0] + bound_box[2], bound_box[1] + bound_box[3]), (225, 0, 0), thickness=2)
        #cv2.imshow('frame', contour_image)
        image_name = str(image_idx).zfill(6) + '.jpg'
        cv2.imwrite(folder_path + image_name, obj_image)
        image_idx += 1


def main():
    image_path = "/home/zs/workspace/ImageProcess/tmp/circle/"
    folder_path = "/home/zs/workspace/ImageProcess/tmp/"
    BatchImageProcess(image_path, folder_path)


# def main():
#     src_Image = cv2.imread("./Images/00001.png")
#     bound_box, contour_image, obj_image = calculatBoundImage(src_Image)
#     print("bound_box", bound_box)
#
#     cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("input image", contour_image)
#
#
#     # 一般源图像进行旋转再提取轮廓
#     rot_image = rotateImage(src_Image, 20, rotate_center=None)
#     cv2.imshow("obj image", obj_image)
#     cv2.imshow("rot image", rot_image)
#     cv2.waitKey(0)
#
#     # vide_file = "./Images/blue_1_82.mp4"
#     # folder_path = "./results/"
#     #
#     # VideotoImage(vide_file, folder_path)


if __name__ == "__main__":
    main()
