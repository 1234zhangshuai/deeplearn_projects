# -*- coding :  utf-8 -*-
# @Data      :  2019-010-17
# @Author    :  zs
# @Email     :  shuaizhang_16@163.com
# @File      :  json_to_dataset.py
# Desctiption:  labelme 标注的生成的 json 转为 maskrcnn 实力分割需要的数据集转换


import argparse
import base64
import json
import os
import os.path as osp
import warnings
import numpy as np

import PIL.Image
import yaml

from labelme import utils
###############################################增加的语句,改下路径即可##############################
import glob
json_list = glob.glob(os.path.join('./maskJson/','*.json'))
cv_mask_dir = "./cv2_mask/"
yaml_dir = "./yaml/"

json_folder = "./keypointJson/"
txt_folder = "./keypoints/"

if not osp.exists(cv_mask_dir):
    os.mkdir(cv_mask_dir)
if not osp.exists(yaml_dir):
    os.mkdir(yaml_dir)
    
if not os.path.exists(txt_folder):
    os.mkdir(txt_folder)
###############################################   end    ##################################


def mask_json2():
    # warnings.warn("This script is aimed to demonstrate how to convert the\n"
    #               "JSON file to a single image dataset, and not to handle\n"
    #               "multiple JSON files to generate a real-use dataset.")

    parser = argparse.ArgumentParser()
    ###############################################  删除的语句  ##################################
    # parser.add_argument('json_file')
    # json_file = args.json_file
    ###############################################    end       ##################################
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    ###############################################增加的语句##################################
    for json_file in json_list:
    ###############################################    end       ##################################

        if args.out is None:
            out_dir = osp.basename(json_file).replace('.', '_')
            img_file = out_dir[:-5]
            out_dir = osp.join(osp.dirname(json_file), out_dir)
        else:
            out_dir = args.out
        #if not osp.exists(out_dir):
            #os.mkdir(out_dir)

        data = json.load(open(json_file))

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        # lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        #
        # label_names = [None] * (max(label_name_to_value.values()) + 1)
        # for name, value in label_name_to_value.items():
        #     label_names[value] = name
        # lbl_viz = utils.draw_label(lbl, img, label_names)
                # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        #captions = ['{}: {}'.format(lv, ln)
        #            for ln, lv in label_name_to_value.items()]
        #lbl_viz = utils.draw_label(lbl, img, captions)
        #PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
        #utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
        #PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
        #with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        #    for lbl_name in label_names:
        #        f.write(lbl_name + '\n')
            
        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        
        #with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        #    yaml.safe_dump(info, f, default_flow_style=False)
        
        utils.lblsave(osp.join(cv_mask_dir, img_file+'.png'), lbl)  # 将掩模保存到cv2_mask文件夹下
        with open(osp.join(yaml_dir, img_file+'.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)         

        print('Saved to: %s' % out_dir)
        
        
def keypoint_json2txt():
    """
    json 文件 中标注点和 lable 保存在 txt 文件
    :param : json_folder json 文件路径
    :param : txt_folder txt 文件保存路径
    """
    json_path = os.listdir(json_folder)
    print(json_path)
    
    for count, json_name in enumerate(json_path):
        jsonname = json_name.split('.')[0]
        json_path = json_folder + json_name
        txt_path = txt_folder + jsonname + '.txt'
        print(json_path, txt_path)
        with open(json_path,'r') as load_f:
            with open(txt_path, 'w') as txt_f:
                load_dict = json.load(load_f)
                #print("load_dict", load_dict)
                value = load_dict['shapes']
                print("value", value[0])
                points = value[0]['points']
                print("points----", points[0])
                
                txt_f.write(str(int(points[0][0])))
                txt_f.write(" ")
                txt_f.write(str(int(points[0][1])))
                txt_f.write(" ")
                txt_f.write(str(2))
                txt_f.write('\n')
                
                # result = str(int(points[0][0])) + " " + str(int(points[0][1])) + " " + str(2) + '\n'
                # txt.write(result)
                
                
def main():
    mask_json2()
    keypoint_json2txt()

if __name__ == '__main__':
    main()

