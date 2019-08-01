#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/11/2019 11:20 AM
# @File : model.py
# @Description :

'''
import glob
import os
import pickle

import keras
from keras import Sequential
from keras.initializers import RandomNormal
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D


def save_model(model, pca, output_file):
    '''

    :param model:
    :param pca:
    :param output_file:
    :return:
    '''
    try:
        with open(output_file, 'wb') as outfile:
            pickle.dump({
                'model': model,
                'pca_fit': pca,
            }, outfile)
        return True
    except:
        return False


def base_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(16, 21, activation='relu', input_shape=input_shape))
    model.add(Conv1D(32, 11, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(model.summary())
    print("CNN Model created.")
    return model


def fully_connected_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=21, input_shape=input_shape, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(Conv1D(32, 11, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(2048, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print("Fully-connected CNN Model created.")
    return model


def without_fully_connected_cnn(input_shape, num_classes):
    model = Sequential()
    activation = 'relu'
    model.add(Convolution1D(2, 9, input_shape=input_shape, activation=activation))
    model.add(BatchNormalization())
    model.add(AveragePooling1D())

    model.add(Convolution1D(2, 7, activation=activation))
    model.add(BatchNormalization())
    model.add(AveragePooling1D())

    model.add(Convolution1D(4, 7, activation=activation))
    model.add(BatchNormalization())
    model.add(AveragePooling1D())

    model.add(Convolution1D(8, 5, activation=activation))
    model.add(BatchNormalization())
    model.add(AveragePooling1D())

    model.add(Convolution1D(12, 3, activation=activation))
    model.add(BatchNormalization())
    model.add(AveragePooling1D())

    model.add(Dropout(0.85, seed=23087))
    model.add(Convolution1D(num_classes, 1))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())

    model.add(Activation('softmax', name='loss'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print("CNN Model created.")
    return model


def build_mlp_architecture(input_shape, num_classes):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    model.add(Dropout(0.5, seed=23087, name='drop1'))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5, seed=23087, name='drop9'))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print("CNN Model created.")
    return model


def load_best_weights(model):
    root_path = os.path.join('weights', 'CNN_Noise_DataAug', model)

    try:
        weight_folds = sorted(next(os.walk(root_path))[1])
    except StopIteration:
        pass

    weights = []
    for fold in weight_folds:
        files_path = os.path.join(root_path, fold, '*.h5')
        cv_weights = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
        weights.append(cv_weights[0])
    return weights
