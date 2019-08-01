#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/16/2019 10:27 AM
# @File : data_info.py
# @Description :

'''
import pandas as pd
from pandas._libs import json

from basic.basic import area_normalization
from load_data_preprocess.load_data import reshape_data, mean_value, plot_mean, plot_std, std_value, plot_mean_std

if __name__ == '__main__':
    print("loading paras: ")
    with open("result/SVM/SVM_MODEL/paras_record.json", 'r') as paras_f:
        paras = json.load(paras_f)
        print(paras)

    Fat = reshape_data('Data/Trainging Data/20%Fat/', X_min=paras['X_min'], X_max=paras['X_max'])[0]
    Tumor = reshape_data('Data/Trainging Data/Tumor/', X_min=paras['X_min'], X_max=paras['X_max'])[0]

    Fat = pd.DataFrame(area_normalization(Fat))
    Tumor = pd.DataFrame(area_normalization(Tumor))

    print('Fat.shape', Fat.shape)
    print('Tumor.shape', Tumor.shape)

    mean_Fat = mean_value(Fat)
    mean_Tumor = mean_value(Tumor)
    plot_mean(mean_Fat, mean_Tumor, path='figures/data_info/Yale/normalization/')

    std_Fat = std_value(Fat)
    std_Tumor = std_value(Tumor)
    plot_std(mean_Fat, mean_Tumor, path='figures/data_info/Yale/normalization/')

    plot_mean_std(mean_Fat, mean_Tumor, std_Fat, std_Tumor,
                  path='figures/data_info/Yale/normalization/')

    # test = reshape_data('Data/Trainging Data/20%Fat/20190614/Fat/S19-20005/Part_1/Cassette_1/',
    #                     X_min=2500, X_max=3200, plot=True)
