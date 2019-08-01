#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/18/2019 10:58 AM
# @File : predictor.py
# @Description :

'''
import pickle

from basic.basic import area_normalization
from load_data_preprocess.load_data import reshape_data


class Predictor(object):
    def __init__(self, model_file):
        with open(model_file, 'rb') as infile:
            self.loaded = pickle.load(infile)
        self.model = self.loaded['model']
        self.pca = self.loaded['pca_fit']

    def read_data(self, path, crop_spectra=True, X_min=800, X_max=2000):
        data = reshape_data(path, X_min=X_min, X_max=X_max)[0]
        if crop_spectra == True:
            data = data.iloc[:, 400:900].values
        data = area_normalization(data)
        X_test_pca = self.pca.transform(data)
        return X_test_pca

    def pca_components_(self):
        return self.pca.components_

    def predict(self, X_test_pca):
        y = self.model.predict(X_test_pca)
        return y

    def decision_function(self, X_test_pca):
        decision = self.model.decision_function(X_test_pca)
        return decision

    def predict_proba(self, X_test_pca):
        proba = self.model.predict_proba(X_test_pca)
        return proba

    def socre(self, x, y):
        score = self.model.score(x, y)
        return score
