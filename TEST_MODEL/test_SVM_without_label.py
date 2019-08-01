#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/16/2019 11:14 AM
# @File : test_SVM.py
# @Description :

'''

import numpy as np
import pandas as pd
from pandas._libs import json

from basic.predictor import Predictor

if __name__ == '__main__':
    predictor = Predictor('../result/SVM/SVM_MODEL/model.pkl')

    # Printing info about model and data
    print("classifier model info: ", predictor.model)
    with open("../result/SVM/SVM_MODEL/paras_record.json", 'r') as paras_f:
        paras = json.load(paras_f)
        print(paras)

    data = predictor.read_data('../Data/Testing Data/2_7-17-2019/20190716/S19-19485',
                               crop_spectra=paras['crop_spectra'],
                               X_min=paras['X_min'], X_max=paras['X_max'])

    labels = predictor.predict(data)
    decision = predictor.decision_function(data)
    proba = np.around(predictor.predict_proba(data), decimals=2)
    print('labels.shape: ', labels.shape, '\n', labels)
    print('decision.shape: ', decision.shape, '\n', decision)
    print('proba.shape: ', proba.shape, '\n', proba)
    result = pd.concat([pd.DataFrame(labels), pd.DataFrame(proba)])
    # result.to_csv("../result/SVM/SVM_TESTING/Yale/labels_proba_1200_noAug.csv")
