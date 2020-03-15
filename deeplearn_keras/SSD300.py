#!/usr/bin/env python
# coding: utf-8

def SSD300(input_shape, num_classes=21):
# Input_shape 为输入的形状（300，300，3）
#num_class 为需要检测的种类
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor
    net['conv1_1'] = Convolution2D(64, 3, 3,
                                  activation='rule',
                                  border_mode='same',
                                  name='conv1_1')(net['input'])
    net['conv1_2'] = Convolution2D(64, 3, 3,
                                  activation='rule',
                                  border_mode='same',
                                  name='conv1_2')(net['conv1_1'])
    net['pool'] = MaxPooling2D((2, 2), strides=(2, 2), 
                                  border_mode='same',
                                  name='pool1')(net['conv1_2'])
    
    #Block 2
    net['conv2_1'] = Convolution2D(128, 3, 3,
                                  activation='relu',
                                  border_model='same',
                                  name='conv2_1')(net['pool1'])
    net['conv2_2'] = Convolution2D(128, 3, 3,
                                  activation='relu',
                                  border_mode='same',
                                  name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2),
                                  border_mode='same',
                                  name='pool2')(net['conv2_2'])
    
    # Block 3
    net['conv3_1'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                   name='pool3')(net['conv3_3'])
    
    # Block 4
    net['conv4_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2),
                                   border_mode='same',
                                   name='pool4')(net['conv4_3'])
    
    # Block 5
    net['conv5_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((2, 2), strides=(2, 2),
                                   border_mode='same',
                                   name='pool5')(net['conv5_3'])
    
    
    # 标红部分就是进行改变的部分，可以看出把FC6换成了空洞卷积，和普通卷积差不多，
    # 就是把一次卷积的感受域扩大了。FC7换成了普通卷积，之后再添加了几个卷积块。
    # FC6
    net['fc6'] = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6),
                                   activation='relu',
                                   border_mode='same',
                                   name='fc6')(net['pool5'])
    # FC7
    net['fc7'] = Convolution2D(1024, 1, 1,
                                   activation='relu',
                                   border_mode='same',
                                   name='fc7')(net['fc6'])
    
    # Block 6
    net['conv6_1'] = Convolution2D(256, 1, 1,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_2'] = Convolution2D(512, 3, 3, subsample=(2, 2),
                                   activation='relu', border_mode='same',
                                   name='conv6_2')(net['conv6_1'])
    
    # Block 7
    net['conv7_1'] = Convolution2D(128, 1, 1, activation='relu',
                                   border_mode='same',
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = Convolution2D(256, 3, 3, subsample=(2, 2),
                                   activation='relu', border_mode='valid',
                                   name='conv7_2')(net['conv7_2'])
    
    # Block 8
    net['conv8_1'] = Convolution2D(128, 1, 1, activation='relu',
                                   border_mode='same',
                                   name='conv8_1')(net['conv7_2'])
    
    net['conv8_2'] = Convolution2D(256, 3, 3, subsample=(2, 2),
                                   activation='relu', border_mode='same',
                                   name='conv8_2')(net['conv8_1'])
    
    # Last Pool
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net[conv8_2])
    
    
'''通过改变后的VGG16得到的多层feature map来预测location 和 confidence。
使用到的feature map 有：conv4_3、fc7、conv6_2、conv7_2、conv8_2、pool6。
总共6层的feature map。因为对于每层的处理步骤差不多，所以就贴出conv4_3处理的代码：'''

# Prediction from conv4_3
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3']) # BN 批归一化
    
    num_priors = 3 #默认框个数
    
    # 对 location 进行卷积预测 
    x = Convolution2D(num_priors * 4, 3, 3,
                         border_mode='same',
                         name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    
    net['conv4_3_norm_mbox_loc'] = x
    
    # 预测完之后进行flatten
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    net['conv4_3_norm_mbox_loc_flat'] = flatten(net['conv4_3_norm_mbox_loc'])
    
    
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
        
    # 进行置信度预测，也就是预测类别
    x = Convolution2D(num_priors * num_class, 3, 3,
                         border_mode='same',
                         name=neme)(net['conv4_3_norm'])
    
    net['conv4_3_norm_mbox_conf'] = x
    
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
    
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2],
                         variances=[0.1, 0.1, 0.2, 0.2],
                         name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])

# 中间其他层类似 conv4_3
    
# Prediction from pool6
# 由于pool6层使用的是globa laverage pool，
# 所以它输出的大小为1*1*256，比较小，不太适合用卷积处理了，
# 就直接用Dense层来处理了
    num_priors = 6
    # 对 location 进行预测
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_loc_flat'] = x
    name = 'pool6_mbox_conf_flat'
    
    if num_classes != 12:
        name += '_{}'.format(num_classes)
    
    # 对置信度进行预测
    x = Dense(num_priors * num_classes, name=name)(net['pool6'])
    net['poo6_mbox_conf_flat'] = x
    
    # 获得默认框
    priorbox = PriorBox(img_size,276.0, max_size=330.0,
                         aspect_ratios=[2, 3],
                         variances=[0.1, 0.1, 0.2, 0.2],
                         name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    net['pool6_reshaped'] = Reshape(target_shape, name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshapeed'])

    # 每层预测完之后，把它们 concatenate 起来
    net['mbox_loc'] = merge([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['pool6_mbox_loc_flat']],
                             mode='concat', concat_axis=1, name='mbox_loc')
    # reshape 维数
    # 计算 default box 的个数
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    
    # location
    net['mbox_loc'] = Reshape((num_boxes, 4),
                              name='mbox_loc_fanal')(net['mbox_loc'])
    # class
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                              name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    
    # merge
    net['prediction'] = merge(net['mbox_loc'],
                              net['mbox_conf'],
                              net['mbox_priorbox'],
                              mode='concat',
                              concat_axis=2,
                              name='predictions')



