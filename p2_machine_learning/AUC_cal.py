#!/usr/bin/env python
# _*_ coding:utf-8 _*_
#
# @Version : 1.0
# @Time    : 2024年12月3日 11:00:29
# @Author  : chenlongxu
# @Mail    : xuchenlong796@qq.com
#
# 描述     : 利用 Numpy 实现 AUC 计算

import numpy as np
import matplotlib.pyplot as plt

def gen_data(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + np.random.uniform(0, 1) * variance
    return x, y

def auc(y_true, y_score):
    auc = 0 
    # NOTE 这里是填写代码的区域

    # END
    return auc

x, y = gen_data(100, 25, 10)
m, n = np.shape(x)
y_true = np.array([1 if i > 50 else 0 for i in range(100)])
y_score = np.random.rand(100)
print(auc(y_true, y_score))
