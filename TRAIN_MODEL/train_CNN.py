#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/11/2019 11:21 AM
# @File : train.py
# @Description :

'''
import numpy as np
from keras.callbacks import ModelCheckpoint
from pandas._libs import json

from basic.basic import load_data_preprocess, training_graphs
from basic.model import fully_connected_cnn, without_fully_connected_cnn, build_mlp_architecture, base_cnn


def train(argv):
    ''' If you want to do data augmentation, please setting aug=True, otherwise, there is no data augmentation

    :param argv:
    :return:
    '''
    # Params

    # Params
    paras = {'train_test_percent': 0.33,
             'crop_spectra': False,
             'aug': False,
             'reshape': True,
             'one_hot': True,
             'X_min': 800,
             'X_max': 2000,
             'n_components': 2,
             'model_path': '../result/CNN/CNN_MODEL/weights/CNN_Noise_DataAug/',
             'num_classes': 2,
             'epochs': 1000,
             'batch_size': 512,
             'seed': 7}

    # Writing paras into .json file
    with open(paras['model_path'] + "paras_record.json", "w") as f:
        json.dump(paras, f)
    print("File successfully written... ...")

    # load dataset and preprocess it, formatting it to a readable tensor
    # Splitting data into training set and testing set
    X_train, X_test, y_train, y_test = load_data_preprocess(paras['train_test_percent'],
                                                            crop_spectra=paras['crop_spectra'],
                                                            aug=paras['aug'],
                                                            reshape=paras['reshape'],
                                                            one_hot=paras['one_hot'],
                                                            X_min=paras['X_min'],
                                                            X_max=paras['X_max'])
    if argv[1] == 'base_cnn':
        model = base_cnn((paras['X_max'] - paras['X_min'], 1), paras['num_classes'])
    if argv[1] == 'fully_cnn':
        model = fully_connected_cnn((paras['X_max'] - paras['X_min'], 1), paras['num_classes'])
    if argv[1] == 'cnn':
        model = without_fully_connected_cnn((paras['X_max'] - paras['X_min'], 1), paras['num_classes'])
    if argv[1] == 'mlp':
        model = build_mlp_architecture((paras['X_max'] - paras['X_min'], 1), paras['num_classes'])

    # fit and run our model
    np.random.seed(paras['seed'])
    best_model_file = \
        "../result/CNN/CNN_MODEL/weights/CNN_Noise_DataAug/" \
        "highest_val_acc_weights_epoch{epoch:02d}-val_acc{val_acc:.3f}_" \
        + str(argv[1]) + ".h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)
    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_test, y_test),
                     nb_epoch=paras['epochs'],
                     batch_size=paras['batch_size'],
                     callbacks=[best_model],
                     shuffle=True,
                     verbose=1)
    print("done training")
    training_graphs(hist)


if __name__ == "__main__":
    argv = ['cnn', 'base_cnn']
    train(argv)
