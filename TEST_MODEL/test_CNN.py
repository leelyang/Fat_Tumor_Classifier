#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/16/2019 2:16 PM
# @File : test_CNN.py
# @Description :

'''
import numpy as np
from keras.engine.saving import load_model
from pandas._libs import json

from load_data_preprocess.load_data import reshape_data

if __name__ == "__main__":
    print("Loading paras... ... ")
    with open("../result/CNN/CNN_MODEL/weights/CNN_Noise_DataAug/paras_record.json", 'r') as paras_f:
        paras = json.load(paras_f)
        print(paras)

    X_test = reshape_data("../Data/Testing Data/S19-19384", X_min=paras['X_min'], X_max=paras['X_max'])[0]
    X_test = np.array(X_test).astype('float32')

    X_test = X_test.reshape(X_test.shape + (1,))
    X_test /= np.max(X_test)

    my_model = load_model(
        '../result/CNN/CNN_MODEL/weights/CNN_Noise_DataAug/highest_val_acc_weights_epoch05-val_acc1.000_base_cnn.h5')
    print(my_model.summary())
    print('X_test.shape: ', X_test.shape)

    result = np.around(my_model.predict(X_test), decimals=2)
    print(result)
    # pd.DataFrame(result).to_csv('../result/CNN/CNN_TESTING/S19-19384/fully_cnn/fully_cnn1200_noAug.csv')
