#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/18/2019 9:13 AM
# @File : D_reduction.py
# @Description :

'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from pandas._libs import json
from sklearn.metrics import classification_report

from basic.basic import make_labels, area_normalization
from basic.predictor import Predictor
from load_data_preprocess.load_data import reshape_data, mean_value

if __name__ == '__main__':
    # Importing data and model
    predictor = Predictor('../result/SVM/SVM_MODEL/model.pkl')

    # Printing info about model and data
    print("classifier model info: ", predictor.model)
    with open("../result/SVM/SVM_MODEL/paras_record.json", 'r') as paras_f:
        paras = json.load(paras_f)
        print(paras)

    data = predictor.read_data('../Data/Testing Data/Yale Data/', crop_spectra=paras['crop_spectra'],
                               X_min=paras['X_min'], X_max=paras['X_max'])
    Fat, Fat_file_path = reshape_data('../Data/Testing Data/Yale Data/Healthy/', X_min=paras['X_min'],
                                      X_max=paras['X_max'])
    Tumor, Tumor_file_path = reshape_data('../Data/Testing Data/Yale Data/Tumor/', X_min=paras['X_min'],
                                          X_max=paras['X_max'])

    path = pd.concat([pd.DataFrame(Fat_file_path), pd.DataFrame(Tumor_file_path)], ignore_index=True)
    path.columns = ['file_path']

    # Getting true label
    true_label = make_labels(Fat, Tumor)
    label = predictor.predict(data)

    print("===================================================================")
    print('1. Model Name:', 'model_svm_Linear_C=0.5')
    print('2. Accuracy： ', predictor.socre(data, true_label))
    target_names = ['Fat', 'Tumor']
    classification_report_logistic = classification_report(true_label, label, target_names=target_names)
    print('3. Classification report:' + '\n', classification_report_logistic)
    print("===================================================================")

    x = data

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # the number of points are 500
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # create a 2D grid figure

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    grid_hat = predictor.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)

    # Plot training set
    ax = plt.figure(figsize=(10, 5), facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
    for n, m in enumerate(np.unique(true_label)):
        plt.scatter(data[true_label == m, 0], data[true_label == m, 1],
                    c=ListedColormap(('g', 'r', 'b'))(n),
                    s=10, cmap=cm_dark)
    plt.legend(['Healthy', 'Tumor'])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('1200_noAug: ' + str(predictor.socre(data, true_label)))
    plt.xlabel('x1', fontsize=10)
    plt.ylabel('x2', fontsize=10)
    plt.grid()

    plt.tight_layout(2)
    plt.subplots_adjust(top=0.92)
    # ax.savefig('../result/SVM/SVM_TESTING/Yale/model_svm_Linear_C=0.5_1200_Aug.png')
    ax.show()

    # ======================================================================================================================#
    # Misclassified data
    choose = input("If you wanna plot misclassified data, press 1: ")
    # choose = '1'
    if (choose == '1'):
        raw_PCA = pd.concat([pd.concat([pd.DataFrame(Fat), pd.DataFrame(Tumor)], axis=0, ignore_index=True),
                             pd.DataFrame(data, columns=['pca1', 'pca2']),
                             pd.DataFrame(true_label, columns=['true_label']),
                             pd.DataFrame(label, columns=['predicted_label'])],
                            axis=1)

        nor_Fat = area_normalization(Fat)
        nor_Tumor = area_normalization(Tumor)
        normalization_PCA = pd.concat([pd.concat(
            [pd.concat([pd.DataFrame(nor_Fat), pd.DataFrame(nor_Tumor)], axis=0, ignore_index=True),
             pd.DataFrame(data, columns=['pca1', 'pca2']),
             pd.DataFrame(true_label, columns=['true_label']),
             pd.DataFrame(label, columns=['predicted_label'])], axis=1), path], axis=1)

        wrong_data = pd.DataFrame()
        for test in range(0, len(label)):
            if (label[test] != true_label[test]):
                print(label[test], true_label[test])
                print(label[test] != true_label[test])
                key = pd.DataFrame(x[test]).T
                wrong_data = pd.concat([wrong_data, key], axis=0)

        if wrong_data.empty:
            print('There is no training data points with wrong labels')
        else:
            wrong_data.columns = ['pca1', 'pca2']
            wrong_data.index = [index for index in range(0, len(wrong_data.iloc[:, 1].values))]
            raw_wrong = pd.merge(raw_PCA, wrong_data, on=['pca1', 'pca2'])

            normalization_wrong = pd.merge(normalization_PCA, wrong_data, on=['pca1', 'pca2'])
            for i in range(0, normalization_wrong.shape[0]):
                if normalization_wrong.iloc[i, -3] == normalization_wrong.iloc[i, -2]:
                    print(i)
                    concat_normalization_wrong = normalization_wrong.drop(normalization_wrong.index[i])
                concat_normalization_wrong = normalization_wrong

            # Plot data using normalization
            for mis in range(0, len(concat_normalization_wrong)):
                ax = plt.figure(figsize=(10, 8), facecolor='w')
                plt.plot(np.arange(paras['X_min'], paras['X_max']),
                         concat_normalization_wrong.iloc[mis, : paras['X_max'] - paras['X_min']].values,
                         label='True label: ' + str(
                             int(concat_normalization_wrong.iloc[mis, -3])) + '\n' + 'File_path: ' + str(
                             concat_normalization_wrong.iloc[mis, -1]))
                plt.plot(np.arange(paras['X_min'], paras['X_max']), mean_value(nor_Fat), label="fat(Nor)-mean")
                plt.plot(np.arange(paras['X_min'], paras['X_max']), mean_value(nor_Tumor), label='tumor(Nor)-mean')
                plt.title('Misclassified(After Normalization): ' + '--' + str(mis))
                plt.legend(bbox_to_anchor=(0, 1.02, 0, .102), loc=3,
                           ncol=1, mode="expand", borderaxespad=0.)
                # ax.savefig(
                #     '../result/SVM/SVM_TESTING/Yale/Misclassified(After Normalization)' + str(
                #         mis) + 'png')
                ax.show()
    pyplot.pause(0)
