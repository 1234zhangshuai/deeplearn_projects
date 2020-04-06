
# Resize images and their corresponding keypoints

from PIL import Image
import os
import numpy as np

mask_root_path = "./cv2_mask/"
mask_save_path = "./cv2_mask/"

img_root_path = "./pic/"
img_save_path = "./pic/"
 
txt_root_path = "./keypoints/"
txt_save_path = "./keypoints/"

mask_name_list = [img_sub_path for img_sub_path in os.listdir(mask_root_path)]
img_name_list = [img_sub_path for img_sub_path in os.listdir(img_root_path)]
txt_name_list = [txt_sub_path for txt_sub_path in os.listdir(txt_root_path)]

for img_name in mask_name_list:
    img_path = mask_root_path + img_name
    img = Image.open(img_path)
    print(img.size)
    resize_img = img.resize((int(img.size[0] / 4), int(img.size[1] / 4)))
    save_path = mask_save_path + img_name
    resize_img.save(save_path)
    print(resize_img.size)
    

for img_name in img_name_list:
    img_path = img_root_path + img_name
    img = Image.open(img_path)
    print(img.size)
    resize_img = img.resize((int(img.size[0] / 4), int(img.size[1] / 4)))
    save_path = img_save_path + img_name
    resize_img.save(save_path)
    print(resize_img.size)
    
    
for txt_name in txt_name_list:
    txt_path = os.path.join(txt_root_path, txt_name)
    print(txt_path)    
    with open(txt_path, "r") as txt_rf:
        points = txt_rf.readline()
    points = [int(point) for point in points.split(" ")]
    print("points: ", points)
    with open(txt_path, "w") as txt_wf:   
        txt_wf.write(str(int(points[0] / 4)))
        txt_wf.write(" ")
        txt_wf.write(str(int(points[1] / 4)))
        txt_wf.write(" 2\n")
        print(str(int(points[0] / 4)), str(int(points[1] / 4)))
        
