#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/11/2019 10:06 AM
# @File : load_data.py
# @Description :
                Importing all data sets
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as ski

from load_data_preprocess.file_operation import get_file_path, extract_cv


# from keras.utils import np_utils


def get_data(filepath):
    ''' Importing data

    :param filepath: String containing a file path to the data you want to import
    :return: DataFrame with data
    '''
    with open(filepath) as f:
        num_cols = max(len(line.split(',')) for line in f)
        f.seek(0)  # start at the head of file
        data = pd.read_csv(f, names=range(num_cols))
    return data


def get_x(data):
    ''' Getting the Wave Number of the original data

    :param data: DataFrame with two columns Wave Number and Fluorescence Corrected
    :return: Wave Number
    '''
    x = data.iloc[:, 0].values
    x = [float(line) for line in x]
    x = np.array(x, dtype=np.float64)
    return x


def get_y(data):
    ''' Getting the Fluorescence Corrected of the original data

    :param data: DataFrame with two columns Wave Number and Fluorescence Corrected
    :return: Fluorescence Corrected
    '''
    y = data.iloc[:, 1].values
    y = [float(line) for line in y]
    y = np.array(y, dtype=np.float64)
    return y


def reshape_data(file_path, X_min, X_max, plot=False):
    ''' Reshaping data into the same range

    :param file_path:
    :param X_min:
    :param X_max:
    :param plot:
    :return:
    '''
    file_path = extract_cv(get_file_path(file_path))
    print(file_path)
    y_set = []
    for path in file_path:
        data = get_data(path).dropna(axis=0)
        data.columns = data.iloc[0, :].values
        data = data[1:]
        data = data[['Wave Number', 'Fluorescence Corrected']]
        x = get_x(data)
        y = get_y(data)
        f = ski.interp1d(x, y, kind="quadratic", fill_value="extrapolate")
        xnew = np.arange(X_min, X_max)
        ynew = f(xnew)
        y_set.append(ynew)
        if plot == True:
            plt.plot(xnew, ynew, linewidth=1)
            plt.show()

    reshape_data = pd.DataFrame(xnew)
    for i in y_set:
        reshape_data = pd.concat([reshape_data, pd.DataFrame(i)], axis=1)

    reshape_data = reshape_data.transpose()
    reshape_data.index = [i for i in range(0, len(reshape_data.iloc[:, 1]))]
    reshape_data = pd.DataFrame(reshape_data.iloc[1:, :].values)
    return reshape_data, file_path


def mean_value(data):
    '''

    :param data: original data
    :return: mean value
    '''
    mean = np.mean(data, axis=0)
    mean = np.array(mean).astype('float32')
    return mean


def std_value(data):
    '''

    :param data:
    :return:
    '''
    std = np.std(data, axis=0)
    std = np.array(std).astype('float32')
    return std


def plot_mean(Fat, Tumor, path='../figures/data_info/'):
    '''

    :param Fat: mean value
    :param Tumor: mean value
    :return:
    '''
    xnew = np.arange(800, 2000)
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 2, 1)
    plt.plot(xnew, Fat, linewidth=3)
    plt.title('Mean_Fat')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(xnew, Tumor, linewidth=3)
    plt.title('Mean_Tumor')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')
    # plt.savefig(path + 'mean_Fat_Tumor.png')
    plt.show()

    plt.figure(figsize=(10, 8))

    plt.plot(xnew, Fat, linewidth=3)
    plt.plot(xnew, Tumor, linewidth=3)
    plt.title('Mean value (Fat and Tumor)')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')
    plt.legend(['mean_Fat', 'mean_Tumor'], loc='upper right')
    # plt.savefig(path + 'mean_Fat_Tumor_Combined.png')

    plt.show()


def plot_std(Fat, Tumor, path='../figures/data_info/'):
    '''

    :param Fat: mean value
    :param Tumor: mean value
    :return:
    '''
    xnew = np.arange(800, 2000)
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 2, 1)
    plt.plot(xnew, Fat, linewidth=3)
    plt.title('std_Fat')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(xnew, Tumor, linewidth=3)
    plt.title('std_Tumor')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')
    # plt.savefig(path + 'std_Fat_Tumor.png')
    plt.show()

    plt.figure(figsize=(10, 8))

    plt.plot(xnew, Fat, linewidth=3)
    plt.plot(xnew, Tumor, linewidth=3)
    plt.title('std value (Fat and Tumor)')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')
    plt.legend(['std_Fat', 'std_Tumor'], loc='upper right')
    # plt.savefig(path + 'std_Fat_Tumor_Combined.png')

    plt.show()


def plot_mean_std(mean_Fat, mean_Tumor, std_Fat, std_Tumor, path='../figures/data_info/'):
    '''

    :param mean_Fat:
    :param mean_Tumor:
    :param std_Fat:
    :param std_Tumor:
    :return:
    '''
    xnew = np.arange(800, 2000)
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 2, 1)
    plt.plot(xnew, mean_Fat, linewidth=3)
    plt.fill_between(xnew, mean_Fat - std_Fat, mean_Fat + std_Fat, alpha=0.5)
    plt.title('mean_std_Fat')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(xnew, mean_Tumor, linewidth=3)
    plt.fill_between(xnew, mean_Tumor - std_Tumor, mean_Tumor + std_Tumor, alpha=0.5)
    plt.title('mean_std_Tumor')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')
    # plt.savefig(path + 'mean_std_Fat_Tumor.png')
    plt.show()

    plt.figure(figsize=(10, 8))

    plt.plot(xnew, mean_Fat, linewidth=3)
    plt.fill_between(xnew, mean_Fat - std_Fat, mean_Fat + std_Fat, alpha=0.5)
    plt.plot(xnew, mean_Tumor, linewidth=3)
    plt.fill_between(xnew, mean_Tumor - std_Tumor, mean_Tumor + std_Tumor, alpha=0.5)
    plt.title('mean_std value (Fat and Tumor)')
    plt.ylabel('Fluorescence Corrected')
    plt.xlabel('Wave Number')
    plt.legend(['mean_Fat', 'mean_Tumor', 'mean_std_Fat', 'mean_std_Tumor'], loc='upper right')
    # plt.savefig(path + 'mean_std_Fat_Tumor_Combined.png')

    plt.show()
