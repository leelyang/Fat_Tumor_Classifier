#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/12/2019 9:25 AM
# @File : basic.py
# @Description :

'''

import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from load_data_preprocess.load_data import reshape_data
from textwrap import wrap

def load_data_preprocess(train_test_percent,
                         crop_spectra=False,
                         aug=False,
                         reshape=False,
                         one_hot=False,
                         X_min=800,
                         X_max=2000,
                         fat_path='../Data/Trainging Data/20%Fat/',
                         tumor_path='../Data/Trainging Data/Tumor/',
                         print_raw=False):
    '''

    :param train_test_percent:
    :param crop_spectra:
    :param aug:
    :param reshape:
    :param one_hot:
    :param X_min:
    :param X_max:
    :param fat_path:
    :param tumor_path:
    :param print_raw:
    :param print_file_path:
    :return:
    '''
    # load data
    Fat, Fat_file_path = reshape_data(fat_path, X_min=X_min, X_max=X_max)
    Tumor, Tumor_file_path = reshape_data(tumor_path, X_min=X_min, X_max=X_max)
    path = pd.concat([pd.DataFrame(Fat_file_path), pd.DataFrame(Tumor_file_path)], ignore_index=True)
    data_All = pd.concat([pd.DataFrame(Fat.append(Tumor, ignore_index=True)), path], axis=1)
    data_All = np.array(data_All)
    print('Fat.shape: ', Fat.shape)
    print('Tumor.shape: ', Tumor.shape)
    print('Data_all.shape: ', data_All.shape)

    labels = make_labels(Fat, Tumor)
    print('labels: ', labels)

    X_train, X_test, y_train, y_test = train_test_split(data_All, labels, test_size=train_test_percent,
                                                        random_state=42)
    X_train_path = X_train
    X_test_path = X_test

    if aug == True:
        from sklearn.decomposition import PCA
        noise_aug = []
        noise = np.copy(X_train)
        mu = np.mean(noise, axis=0)
        pca = PCA()
        noise_model = pca.fit(noise)
        nComp = 10
        Xhat = np.dot(pca.transform(noise)[:, :nComp], pca.components_[:nComp, :])
        noise_level = np.dot(pca.transform(noise)[:, nComp:], pca.components_[nComp:, :])
        Xhat += mu
        SNR = np.linspace(1, 5, 50)
        for i in range(len(SNR)):
            noise_aug.append(SNR[i] * noise_level + Xhat)
            j = 0
            for spectra in noise_aug[i]:
                noise_aug[i][j] = spectra / np.max(spectra)
                j += 1
        X_train = np.array(noise_aug).reshape(50 * 177, 1200)
        y_train = [item for i in range(50) for item in y_train]

    if crop_spectra == True:
        X_train, X_test = crop(X_train, X_test, min=400, max=900)

    X_train, X_test, y_train, y_test = preprocess(X_train[:, :-1], X_test[:, :-1], y_train, y_test, reshape=reshape,
                                                  mean_center=False,
                                                  norm=True,
                                                  one_hot=one_hot)
    print('X_train.shape: ', X_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_train.shape: ', y_train.shape)
    print('y_test.shape: ', y_test.shape)
    if print_raw == True:
        return X_train, X_test, y_train, y_test, X_train_path, X_test_path
    return X_train, X_test, y_train, y_test


def make_labels(Fat, Tumor):
    ''' Adding labels

    :param Fat:
    :param Tumor:
    :return:
    '''
    labels = []
    for i in range(len(Fat)):
        labels.append(0)
    for i in range(len(Tumor)):
        labels.append(1)
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(labels)
    return labels


def crop(X_train, X_test, min=400, max=900):
    ''' Choosing part of features

    :param X_train:
    :param X_test:
    :param min:
    :param max:
    :return:
    '''
    crop_X_train = X_train[:, min:max]
    crop_X_test = X_test[:, min:max]
    return crop_X_train, crop_X_test


def preprocess(X_train, X_test, y_train, y_test, mean_center=True, norm=True, reshape=True, one_hot=True):
    '''

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param mean_center:
    :param norm:
    :param reshape:
    :param one_hot:
    :return:
    '''
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')

    if norm == True:
        if reshape == True:
            # X_train = np.expand_dims(X_train, axis=2)
            # X_test = np.expand_dims(X_test, axis=2)
            X_train = area_normalization(X_train)
            X_test = area_normalization(X_test)
            X_train = X_train.reshape(X_train.shape + (1,))
            X_test = X_test.reshape(X_test.shape + (1,))
        else:
            # X_train = np.transpose(preprocessing.MaxAbsScaler().fit_transform(np.transpose(X_train)))
            # X_test = np.transpose(preprocessing.MaxAbsScaler().fit_transform(np.transpose(X_test)))
            X_train = area_normalization(X_train)
            X_test = area_normalization(X_test)
            # X_train = max_normalization(X_train)
            # X_test = max_normalization(X_test)
        print('Data normalized')
    if mean_center == True:
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)
        print('Data mean-centered')

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    if one_hot == True:
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        print('Data one-hot encoded')

    print("Total of " + str(len(np.unique(y_test))) + " classes.")
    print("Total of " + str(len(X_train)) + " training samples.")
    print("Total of " + str(len(X_test)) + " testing samples.")

    return X_train, X_test, y_train, y_test


def training_graphs(hist):
    '''

    :param hist:
    :return:
    '''
    # summarize history for accuracy
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'], linewidth=3)
    plt.title('Model Training Accuracy')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Epoch')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], linewidth=3)
    plt.title('Model Training Loss')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epoch')
    # plt.savefig('result/CNN/CNN_FIGURE/fully_cnn/training_accuracy_fully_cnn_500_Aug.png')
    plt.show(block=False)

    plt.figure(figsize=(10, 8))

    plt.plot(hist.history['val_acc'], linewidth=3)
    plt.plot(hist.history['acc'], linewidth=3)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Train'], loc='lower right')
    # plt.savefig('result/CNN/CNN_FIGURE/fully_cnn/test_accuracy_fully_cnn_500_Aug.png')
    plt.show(block=False)


def area_normalization(X):
    '''

    :param X:
    :return:
    '''
    X = pd.DataFrame(X).iloc[:].values
    sum = []
    temp = 0
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            temp += X[i][j]
        sum.append(temp)
        temp = 0

    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X[i][j] = X[i][j] / sum[i]

    return 100 * X


def plot_PCA(projected, target, pc1, pc2):
    '''

    :param projected:
    :param target:
    :param pc1:
    :param pc2:
    :return:
    '''
    plt.figure()
    plt.scatter(projected[:, pc1], projected[:, pc2],
                c=target, edgecolor='none', alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show(block=False)


def plot_PCA_components(n_components, X):
    '''

    :param n_components:
    :param X:
    :return:
    '''
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)

    plt.figure()
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(60, 100)
    plt.style.context('seaborn-whitegrid')
    plt.plot(var)
    plt.show(block=False)
    print(var)


def plot_PCA_interactive(projected, target, key, spectra, pc1, pc2, xmin, xmax):
    fig = plt.figure()
    plt.scatter(projected[:, pc1], projected[:, pc2],
                c=target, edgecolor='none', alpha=0.5)
    plt.xlabel('component ' + str(pc1+1))
    plt.ylabel('component ' + str(pc2+1))
    plt.colorbar()
    plt.show(block=False)

    def onclick(event):
        min_distance = sys.float_info.max
        min_key = 0
        for i in range(0, len(key)):
            distance = (event.xdata - key.iloc[i, pc1])**2 + (event.ydata - key.iloc[i, pc2])**2
            if distance < min_distance:
                min_distance = distance
                min_key = i
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f path=%s' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata, key.iloc[min_key, -1]))
        newfig = plt.figure()
        plt.title("\n".join(wrap(key.iloc[min_key, -1], 60)))
        plt.plot(range(xmin, xmax), spectra.iloc[min_key, :])
        plt.show(block=False)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, cid


def plot_embedding(data, label, title):
    '''

    :param data:
    :param label:
    :param title:
    :return:
    '''
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig