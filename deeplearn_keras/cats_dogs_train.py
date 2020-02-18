#!/usr/bin/env python
# coding: utf-8

# # 用卷积神经网络训练猫狗数据集,采用了数据增强

from keras import layers
from keras import models
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image # 图像预处理工具的模块

import os
import matplotlib.pyplot as plt

base_dir = 'E:/deeptest/Deeplearn_keras/python_keras/cats_and_dogs_small' # 数据集的目录(已由data_dir.py生成)
############################################################################
# 训练集目录
train_dir = os.path.join(base_dir, 'train')
# 验证集目录
validation_dir = os.path.join(base_dir, 'validation')
# 测试集目录
test_dir = os.path.join(base_dir, 'test')

# 猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
# 狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')

# 猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
# 狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
# 狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')

#卷积网络模型实例化
model = models.Sequential()

#第1个卷积层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

#第2个卷积层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#第3个卷积层
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#第4个卷积层
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#平铺，展平数据
model.add(layers.Flatten())

#添加1个Dropout层
model.add(layers.Dropout(0.5))

#添加2个Dense层
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#配置模型用于训练
model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=1e-4), 
             metrics=['acc'])

#使用ImageDataGenerator从目录中读取图像

# 利用数据增强生成器训练卷积神经网络
train_datagen = ImageDataGenerator(
                    rescale=1./255, 
                    rotation_range=40, 
                    width_shift_range=0.2, 
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
                    
# 注意，不能增强验证数据
test_datagen = ImageDataGenerator(rescale=1./255) 

# 使用ImageDataGenerator从目录中读取图像
train_generator = train_datagen.flow_from_directory(
                    train_dir, # 目标目录
                    target_size=(150, 150), # 将所有图像的大小调整为150 x 150
                    batch_size=32, 
                    class_mode='binary') # 因为使用了binary_crossentropy损失，所以需要用二进制标签

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='binary')

#利用批量生成器拟合模型
history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

#保存模型
model.save('cats_and_dos_small_1.h5')

#绘制 训练过程中的损失曲线和精度曲线

#纵坐标数据
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

#横坐标
epotchs = range(1, len(acc) + 1)

plt.plot(epotchs, acc, 'bo', label='Training acc')
plt.plot(epotchs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epotchs, loss, 'bo', label='Training loss')
plt.plot(epotchs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()





