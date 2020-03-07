# -*- coding :  utf-8 -*-
# @Data      :  2019-08-17
# @Author    :  zs
# @Email     :  shuaizhang_16@163.com
# @File      :  ImageDataSetGeneration.py
# Desctiption:  生成物体分类的图像数据集

import numpy as np
import cv2
import os
from lxml import etree, objectify



def rotateImage(src_image, rotate_deg):
    """
    对图像进行旋转
    :param src_image: 输入源图像
    :param rotate_dog: 旋转角度
    :return: 旋转后的图像
    """
    img_h, img_w = src_image.shape[0:2]
    rotate_mat = cv2.getRotationMatrix2D((img_w / 2.0, img_h / 2.0), rotate_deg, 1.0)
    dst_image = cv2.warpAffine(src_image, rotate_mat, (img_w, img_h))
    return dst_image


def calculateBoundImage(src_image):
    """
    求图像中物体的边界矩形
    :param src_image: 源图像
    :return: 图像中物体的边界矩形、轮廓图、目标图像
    """

    tmp_image = src_image.copy()
    if len(tmp_image.shape) == 3:
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
    ret, thresh_images = cv2.threshold(tmp_image, 0, 255,cv2.THRESH_BINARY)
    contours_ls, _ = cv2.findContours(thresh_images, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 所有轮廓的边界框
    all_points = np.concatenate(contours_ls, axis=0)
    bound_box = cv2.boundingRect(all_points)
    return bound_box


def randomMoveObjectInImage(src_image, src_bound_box):
    """
    将物体在图像中随机摆放
    :param src_image: 背景图 COCO/VOC
    :param src_bound_box: 原始边界框
    :return: 相机旋转后的边界框
    """
    x, y, w, h = src_bound_box
    img_h, img_w = src_image.shape[0:2]
    
    # 为了防止随机产生的矩形框在图像之外，
    # 即使得以左上角坐标点画的矩形框在图像之内
    img_h -= h
    img_w -= w
    # 随机生成0到1之间的两个数
    random_array = np.random.uniform(0.0, 1.0, 2)
    # 随机产生左上角的坐标值
    bbox_x = np.int(img_w * random_array[0])
    bbox_y = np.int(img_h * random_array[1])
    return np.array([bbox_x, bbox_y, w, h])


def calculateIOU(bound_box_1, bound_box_2):
    """
    计算两个 bound_box 之间的 IOU
    :param bound_box_1: 边界框 1, shape [x, y, w, h]
    :param bound_box_2: 边界框 2，shape [x, y, w, h]
    :return: 两个 bound box 之间的 IOU 值
    """

    min_xy = np.maximum(bound_box_1[0:2], bound_box_2[0:2])
    max_xy = np.minimum(bound_box_1[0:2] + bound_box_2[2:4],
                        bound_box_2[0:2] + bound_box_2[2:4])

    delta_xy = max_xy - min_xy
    # 求相交重叠的部分的面积，交集
    intersection_area = delta_xy[0] * delta_xy[1]
    if (intersection_area < 0):
        return
    # 求并集的面积  
    box_area_1 = bound_box_1[2] * bound_box_1[3]
    box_area_2 = bound_box_2[2] * bound_box_2[3]

    union_area = box_area_1 + box_area_2 - intersection_area
    return intersection_area / union_area


def resizeObjectImage(src_image, max_min_box_size):
    """
    对物体图像进行随机缩放
    :param src_image: 原始图像
    :param max_min_box_size: 缩放后图像中的物体的 bound box 的最大边的范围
    :return: 缩放后的图像
    """
    src_bbox = calculateBoundImage(src_image)
    # 选取原图像的最大边框
    src_bbox_max = np.max(src_bbox[2:4])
    cur_bbox_max = np.random.uniform(max_min_box_size[1], max_min_box_size[0], 1)[-1]
    # 缩放比
    cur_ratio = cur_bbox_max / src_bbox_max

    src_h, src_w = src_image.shape[0:2]
    dst_h, dst_w = np.int(src_h * cur_ratio), np.int(src_w * cur_ratio)
    dst_image = cv2.resize(src_image, (np.int(dst_w), np.int(dst_h)))
    return dst_image


def addObjectToImage(backgroup_image, obj_image, bound_box):
    """
    将目标物体添加到背景图中（贴图）
    :param backgroup_image: 背景图
    :param obj_image: 目标物体图
    :param bound_box: 边界矩形框
    :return: 添加了目标物体的背景图
    """

    tmp_image = obj_image.copy()
    # RGB图要转换为灰度图
    if len(tmp_image.shape) == 3:
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
        
    # 设置目标图大于5的像素，由于是掩码之后的图所以只剩下目标物
    mask = tmp_image > 5
    
    # 矩形框左上和右下角坐标
    min_x, min_y, max_x, max_y = bound_box[0], bound_box[1], bound_box[0] + bound_box[2], bound_box[1] + bound_box[3]
    
    # 将目标物体添加到到背景图（替换其相应像素）中，
    # 其位置是背景图像中的矩形框内像素位置
    backgroup_image[np.int(min_y):np.int(max_y), np.int(min_x):np.int(max_x)][mask] = obj_image[mask]
    return backgroup_image


def formImageAndlabel(background_image, obj_ls, max_min_size_ration, iou_thres):
    """
    形成训练图像，并生成对应的 label 列表
    :param background_image: 输入背景图
    :param obj_ls: 目标 list
    :param max_min_size_ration: 最大最小旋转角度
    :param iou_thres: IOU 阈值
    :return: 返训练的图像，对应的 label
    """
    
    # 设置旋转的角度
    max_ratio, min_ratio = max_min_size_ration
    
    image_size = np.min(background_image.shape[0:2])
    dst_image = background_image.copy()
    
    max_min_box_size = [np.int(max_ratio * image_size), np.int(min_ratio * image_size)]
    label_ls = []
    for obj_image, obj_name in obj_ls:
        # 对目标图像进行随机缩放
        resize_obj_image = resizeObjectImage(obj_image, max_min_box_size)
        # 对目标图像进行随机旋转
        rotate_image = rotateImage(resize_obj_image, np.random.uniform(0, 360, 1)[-1])
        # 多次迭代, 直到将图像平移到适当位置为止
        # 获取变换后图像的矩形框数据
        src_bbox = calculateBoundImage(rotate_image)
        
        # 获得边界框包围的目标图像
        sub_obj_image = rotate_image[src_bbox[1]:src_bbox[1] + src_bbox[3], src_bbox[0]:src_bbox[0] + src_bbox[2]]
        iter_cnt = 100
        if len(label_ls) == 0:
            iter_cnt = 1
        for iter_idx in range(iter_cnt):
            # 将上述变换之后的目标图像矩阵框随机放到背景图片上
            dst_bbox = randomMoveObjectInImage(dst_image, src_bbox)
            if len(label_ls) != 0:
                is_fit = True
                for tmp_box, tmp_obj_name in label_ls:
                    #print("....", tmp_box)
                    #print("+++++", dst_bbox)
                    # 此时是再一次变换后的矩形框位置，设置IoU是为了避免两次变换后的矩形框的距离太近
                    IOU = calculateIOU(tmp_box, dst_bbox)           
                    if (IOU is not None) and (IOU > iou_thres):
                        is_fit = False
                        break
                if is_fit == False:
                    continue
                else:
                    break
        # 得到的贴图后的图像
        dst_image = addObjectToImage(dst_image, sub_obj_image, dst_bbox)
        # 添加到储存到标签列表中
        label_ls.append([dst_bbox, obj_name])
    return dst_image, label_ls


def formImageLableXML(src_image, image_file_name, label_info, label_path):
    """
    生成图片的 label XML
    :param src_image: 原始图像
    :param image_file_name: 图像的文件名
    :param label_infor: 标签信息
    :param label_path: 标签的路径
    :return: XML
    """

    ele = objectify.ElementMaker(annotate=False)
    anno_tree = ele.annotation(
        ele.folder('VOC2012'),
        ele.filename(image_file_name),
        ele.source(
            ele.database('The VOC2012 Database'),
            ele.annotation('PASCAL VOC2012'),
            ele.image('flickr'),
            ele.flickrid('264265361')
            ),
        ele.owner(
            ele.flickrid('zs'),
            ele.name('zs')
        ),
        ele.size(
            ele.width(str(src_image.shape[0])),
            ele.height(str(src_image.shape[1])),
            ele.depth(str(src_image.shape[2]))
        ),
        ele.segmented('0')
    )
    for cur_box, cur_obj_name in label_info:
        cur_ele = objectify.ElementMaker(annotate=False)
        cur_tree = cur_ele.object(
            ele.name(cur_obj_name),
            ele.pose('Frontal'),
            ele.truncated('0'),
            ele.difficult('0'),
            ele.bndbox(
                ele.xmin(str(int(cur_box[0]))),
                ele.ymin(str(int(cur_box[1]))),
                ele.xmax(str(int(cur_box[0] + cur_box[2]))),
                ele.ymax(str(int(cur_box[1] + cur_box[3])))
            )
        )
        anno_tree.append(cur_tree)
    etree.ElementTree(anno_tree).write(label_path, pretty_print=True)


def main():
    obj_name_ls = ['Oreo', 'PacificBiscuit']
    # 各种物体对应的图像的路径
    base_obj_file_name = 'E:/deeplearn/keras-yolov3/DataPrepropress/originImage/'
    obj_file_name = [base_obj_file_name + cur_obj for cur_obj in obj_name_ls]
    print(obj_file_name)
    # 每个种类的样本数量
    obj_count = 350

    # 图像中物体出现最多的次数
    image_max_obj_cnt = 3

    # 图像中物体的 bound box 的最大尺寸点，整个图像最小尺寸比例，
    max_size_radio = 0.45
    min_size_radio = 0.20

    # 图像的总数
    image_count = len(obj_name_ls) * obj_count

    # 数据集的保存路径
    dataset_basic_path = 'E:/deeplearn/keras-yolov3/DataPrepropress/VOC/VOCdevkit/VOC2012/'
    image_folder = dataset_basic_path + 'JPEGImages/'  # 保存生成后图像
    # print(image_folder)
    label_folder = dataset_basic_path + 'Annotations/'  # 保存生成后的xml文件
    # print(label_folder)
    image_set_folder = dataset_basic_path + 'ImageSets/Main/' # 保存划分后的train.txt test.txt val.txt （保存内容为每张图片的数字号）
    # print(image_set_folder)

    for data_idx in range(image_count):
        # 获取 VOC 数据集中图像文件夹中所有文件的名称 # 背景图片存放路径
        voc_folder_dir = 'E:/deeplearn/keras-yolov3/DataPrepropress/backgroundImage/'
        voc_image_file_list = os.listdir(voc_folder_dir) # 背景路径下所有背景图片名称
        # 获取物体图像的文件名列表
        obj_image_ls_ls = []
        for obj_image_dir in obj_name_ls:
            cur_image_dir = base_obj_file_name + obj_image_dir  # 已有数据的路径 比如../circle ../square
            obj_image_ls_ls.append(os.listdir(cur_image_dir))  # os.listdir(path)该路径下所有文件和文件夹名称

        # 随机取一张 VOC 图做背景
        background_image_filename = voc_image_file_list[np.random.randint(0, len(voc_image_file_list), 1)[-1]]  # numpy.random.randint(low,high=None,size=None,dtype) 
        background_image_filepath = voc_folder_dir + os.sep + background_image_filename # 选取的背景的路径+文件名
        background_image = cv2.imread(background_image_filepath)  # 1.背景图*****************

        # 随机取若干物体
        obj_image_name_ls = []  # 保存所选物体的图片和对应的类别
        obj_cnt = np.random.randint(1, image_max_obj_cnt, 1)[-1]  # 图像中物体出现最多的次数 image_max_obj_cnt
        for obj_idx in range(obj_cnt):
            cur_obj_idx = np.random.randint(0, len(obj_image_ls_ls), 1)[-1]  # 根据现有物体图片的数量得到随机的id
            cur_obj_image_ls = obj_image_ls_ls[cur_obj_idx]  # 根据id挑选出对应的图片所属类别
            cur_obj_file = cur_obj_image_ls[np.random.randint(0, len(cur_obj_image_ls), 1)[-1]]  # 从该类别中随机选取一张图片
            cur_obj_image = cv2.imread(base_obj_file_name + obj_name_ls[cur_obj_idx] + os.sep + cur_obj_file) # 2.当前选择的图片***************
            obj_image_name_ls.append([cur_obj_image, obj_name_ls[cur_obj_idx]])

        # ************---------------------****************
        # 随机生成图像（输入参数：背景图， 随机选取的物体图列表， box缩放大小， iou阈值）
        get_image, label_ls = formImageAndlabel(background_image, obj_image_name_ls, [max_size_radio, min_size_radio], iou_thres=0.05)
        
        # # 保存图像与标签
        cur_image_name = str(data_idx).zfill(6) + '.jpg'  # 当前文件名根据序号扩充为6位数
        # print(cur_image_name)
        cur_label_name = str(data_idx).zfill(6) + '.xml'  # 对应的标签名（Annotations/路径下）
        # print(cur_label_name)

        cv2.imwrite(image_folder + cur_image_name, get_image)  # 保存图片于JPEGImages/目录下
        
        # ************---------------------****************
        # 保存对应的xml文件与Annotations/目录下
        formImageLableXML(get_image, cur_image_name, label_ls, label_folder + cur_label_name)  
    
        # label_ls中包含了box信息和对应的类别, 在图中画出边框
        for obj_bbox, obj_name in label_ls:
            pnt_1 = tuple(map(int, obj_bbox[0:2]))
            pnt_2 = tuple(map(int, (obj_bbox[0:2] + obj_bbox[2:4])))
            cv2.rectangle(get_image, pnt_1, pnt_2, (0, 0, 255))
        print(cur_image_name)  # 打印出当前文件的文件名 六位数+jpg example:000003.jpg

        cv2.imshow("get image", get_image)
        cv2.waitKey(5)
    train_set_name = 'train.txt'
    train_val_name = 'val.txt'
    test_set_name = 'test.txt'
    idx_thre = np.int(0.6 * image_count)  # 0.6乘以总的图片生成数
    idx_thre_ = np.int(0.8 * image_count) # 按比例将数据分为训练集、验证集和测试集

    # Main/目录下的三个文件
    train_file = open(image_set_folder + train_set_name, 'w')  # 保存train.txt
    for line_idx in range(idx_thre):
        line_str = str(line_idx).zfill(6) + '\n'
        train_file.write(line_str);
    train_file.close()

    train_val_file = open(image_set_folder + train_val_name, 'w')  # 保存val.txt
    for line_idx in range(idx_thre, idx_thre_):
        line_str = str(line_idx).zfill(6) + '\n'
        train_val_file.write(line_str)
    train_val_file.close()

    test_file = open(image_set_folder + test_set_name, 'w')  # 保存test.txt
    for line_idx in range(idx_thre_, image_count):
        line_str = str(line_idx).zfill(6) + '\n'
        test_file.write(line_str)
    test_file.close()


if __name__ == '__main__':
    main()

