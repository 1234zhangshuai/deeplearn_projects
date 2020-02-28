#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# x为初始输入值
x1 = 0.1
x2 = 0.88

# y为输出值（真值）
y1 = 0.55
y2 = 1

w1,w2,w3,w4,w5,w6,w7,w8 = 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0

# 迭代次数
iter = 1
for i in range(iter):
    # 开始计算反向传播
    h_in_1 = w1*x1 + w2*x2
    h_in_2 = w3*x1 + w4*x2

    # h_in为传播到隐藏层的值
    print('h_in_1 = %f,h_in_2 = %f'%(h_in_1,h_in_2))
    
    h_out_1 = sigmoid(h_in_1)
    h_out_2 = sigmoid(h_in_2)
    
    # h_out从隐藏层传出的值
    print('h_out_1 = %f,h_out_2 = %f'%(h_out_1,h_out_2))
    
    o_in_1 = w5*h_out_1 + w6*h_out_2
    o_in_2 = w7*h_out_1 + w8*h_out_2
    
    # o_in为传入输出层的值
    print('o_in_1 = %f,o_in_2 = %f'%(o_in_1,o_in_2))
    
    o_out_1 = sigmoid(o_in_1)
    o_out_2 = sigmoid(o_in_2)
    
    # o_out为输出的结果
    print('o_out_1 = %f,o_out_2 = %f'%(o_out_1,o_out_2))
    
    # 计算loss

    lost1 = (y1 - o_out_1)**2/2
    lost2 = (y2 - o_out_2)**2/2
    lost = lost1 + lost2
    print('lost = %f'%(lost))
    
    # 反向传播求梯度
    diff_w5 = -(y1 - o_out_1)*o_out_1*(1 - o_out_1)*h_out_1
    diff_w6 = -(y1 - o_out_1)*o_out_1*(1 - o_out_1)*h_out_2
    diff_w7 = -(y2 - o_out_2)*o_out_2*(1 - o_out_2)*h_out_1
    diff_w8 = -(y2 - o_out_2)*o_out_2*(1 - o_out_2)*h_out_2
    
    diff_w1 = (-(y1-o_out_1)*o_out_1*(1-o_out_1)*w5 - (y2-o_out_2)*o_out_2*(1-o_out_2)*w7)*(1-h_out_1)*h_out_1*x1
    diff_w2 = (-(y1-o_out_1)*o_out_1*(1-o_out_1)*w5 - (y2-o_out_2)*o_out_2*(1-o_out_2)*w7)*(1-h_out_1)*h_out_1*x2
    diff_w3 = (-(y2-o_out_2)*o_out_2*(1-o_out_2)*w6-(y1-o_out_1)*o_out_1*(1-o_out_1)*w8)*(1-h_out_2)*h_out_2*x1
    diff_w4 = (-(y2-o_out_2)*o_out_2*(1-o_out_2)*w6-(y1-o_out_1)*o_out_1*(1-o_out_1)*w8)*(1-h_out_2)*h_out_2*x2
    
    print('diff_w5 = %f, diff_w6 = %f, diff_w7 = %f, diff_w8= %f'%(diff_w5, diff_w6, diff_w7, diff_w8))
    print('diff_w1 = %f, diff_w2 = %f, diff_w3 = %f, diff_w4 = %f'%(diff_w1, diff_w2, diff_w3, diff_w4))

    # diff_w为w对代价函数的偏导数
    theta = 0.5
    updata_w5 = w5 - theta*diff_w5
    updata_w6 = w6 - theta*diff_w6
    updata_w7 = w7 - theta*diff_w7
    updata_w8 = w8 - theta*diff_w8
    updata_w1 = w1 - theta*diff_w1
    updata_w2 = w2 - theta*diff_w2
    updata_w3 = w3 - theta*diff_w3
    updata_w4 = w4 - theta*diff_w4
    
    print('update_w5 = %f, update_w6 = %f, update_w7 =%f, update_w8 = %f'%(updata_w5,updata_w6,updata_w7,updata_w8))
    print('update_w1 = %f, update_w2 = %f, update_w3 =%f, update_w4 = %f'%(updata_w1,updata_w2,updata_w3,updata_w4))

    # 将更新的梯度赋值给下一次迭代的权重
    w5 = updata_w5
    w6 = updata_w6
    w7 = updata_w7
    w8 = updata_w8
    w1 = updata_w1
    w2 = updata_w2
    w3 = updata_w3
    w4 = updata_w4
    
print(lost)
print(o_out_1, o_out_2)


# In[ ]:




