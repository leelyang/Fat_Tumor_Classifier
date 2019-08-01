#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/12/2019 10:47 AM
# @File : train_SVM.py
# @Description :

'''
import os
from time import time
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pandas._libs import json
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

from basic.basic import load_data_preprocess, plot_PCA_components, plot_PCA, \
    area_normalization, plot_embedding, plot_PCA_interactive
from basic.model import save_model
from load_data_preprocess.load_data import mean_value, reshape_data

if __name__ == '__main__':
    # Params
    paras = {'train_test_percent': 0.33,
             'crop_spectra': False,
             'aug': False,
             'reshape': False,
             'one_hot': False,
             'X_min': 2700,
             'X_max': 3100,
             'n_components': 2,
             'model_path': '../result/SVM/SVM_MODEL/',
             'print_raw': True,
             'show_interactive_plot': True,
             'show_pca_loadings': False,
             'show_tsne': False}

    plt.ioff()
    # Writing paras into .json file
    with open(paras['model_path'] + "paras_record.json", "w") as f:
        json.dump(paras, f)
    print("File successfully written... ...")

    nor_Fat = area_normalization(
        reshape_data('../Data/Trainging Data/20%Fat/', X_max=paras['X_max'], X_min=paras['X_min'])[0])
    nor_Tumor = area_normalization(
        reshape_data('../Data/Trainging Data/Tumor/', X_max=paras['X_max'], X_min=paras['X_min'])[0])

    # Splitting data into training set and testing set
    X_train, X_test, y_train, y_test, X_train_path, X_test_path = load_data_preprocess(
        paras['train_test_percent'],
        crop_spectra=paras['crop_spectra'],
        aug=paras['aug'],
        reshape=paras['reshape'],
        one_hot=paras['one_hot'],
        X_min=paras['X_min'],
        X_max=paras['X_max'],
        print_raw=paras['print_raw'],
        fat_path='../Data/Testing Data/Yale Data/Healthy/',
        tumor_path='../Data/Testing Data/Yale Data/Tumor/')

    path = pd.DataFrame(pd.concat(
        [pd.DataFrame(X_train_path[:, -1]), pd.DataFrame(X_test_path[:, -1])],
        ignore_index=True).iloc[:, -1].values)
    data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)
    data_y = np.concatenate([y_train, y_test])

    # Adding Yale's data
    # X_train, X_test, y_train, y_test = \
    #     load_data_preprocess(train_test_percent, crop_spectra=False, aug=False, reshape=False, one_hot=False,
    #                          fat_path='../Data/Testing Data/Yale Data/Healthy',
    #                          tumor_path='../Data/Testing Data/Yale Data/Tumor')
    # X_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_train_yale)]).iloc[:].values
    # X_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(X_test_yale)]).iloc[:].values
    # y_train = np.append(y_train, y_train_yale)
    # y_test = np.append(y_test, y_test_yale)

    # Applying dimensional reduction
    t0 = time()
    pca = PCA(n_components=paras['n_components'], svd_solver='randomized', whiten=True)
    print("done in %0.3fs" % (time() - t0))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
#    plot_PCA(X_train_pca, y_train, 0, 1)  # You can choose which two pcs you want to plot

    if paras['show_tsne']:
        # Applying TSNE
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        data_tsne = tsne.fit_transform(data)
        label = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], ignore_index=True)
        label = label.iloc[:, -1].values
        fig = plot_embedding(data_tsne, label,
                             't-SNE embedding of the data (time %.2fs)'
                             % (time() - t0))
        plt.show(block=False)

    if paras['show_interactive_plot']:
        data_pca = pca.transform(data.to_numpy())
        pca_to_path_key = pd.concat([pd.DataFrame(data_pca), path], axis=1)
        fig_interactive, cid_interactive = plot_PCA_interactive(data_pca, data_y, pca_to_path_key, data,
                                                                0, 1, paras['X_min'], paras['X_max'])

    if paras['show_pca_loadings']:
        plot_PCA_components(50, X_train)
        for i in range(len(pca.components_)):
            plt.figure()
            plt.title('PCA ' + str(i))
            plt.plot(range(paras['X_min'], paras['X_max']), pca.components_[i])
            plt.show(block=False)

    x = np.array(pd.concat([pd.DataFrame(X_train_pca[:, :2]), pd.DataFrame(X_test_pca[:, :2])]))

    # different classifier
    clfs = [svm.SVC(C=0.5, kernel='linear', probability=True)]

    titles = ['Linear_C=0.5']
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # the number of points are 500
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # create a 2D grid figure

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])

    for i, clf in enumerate(clfs):
        model = clf.fit(X_train_pca, y_train.ravel())

        y_pred = clf.predict(X_test_pca)
        print("===================================================================")
        print('1. Model Name:', titles[i])
        print('2. Accuracy： ', clf.score(X_test_pca, y_test))
        target_names = ['Fat', 'Tumor']
        classification_report_logistic = classification_report(y_test, y_pred, target_names=target_names)
        print('3. Classification report:' + '\n', classification_report_logistic)
        print("===================================================================")
        save_model(clf, pca, os.path.join(paras['model_path'], 'model.pkl'))

        if paras['n_components'] != 2:
            print("Cannot plot more than 2-D features.")
        else:
            ax = plt.figure(figsize=(10, 5), facecolor='w')
            grid_hat = clf.predict(grid_test)
            grid_hat = grid_hat.reshape(x1.shape)

            # Plot training set
            plt.subplot(1, 2, 1)
            plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
            for n, m in enumerate(np.unique(y_train)):
                plt.scatter(X_train_pca[y_train == m, 0], X_train_pca[y_train == m, 1],
                            c=ListedColormap(('g', 'r', 'b'))(n),
                            s=10, cmap=cm_dark)
            # plt.legend(['Fat', 'Tumor'])
            plt.xlim(x1_min, x1_max)
            plt.ylim(x2_min, x2_max)
            plt.title('Training visualization')
            plt.suptitle(u'SVM: ' + titles[i])
            plt.xlabel('x1', fontsize=10)
            plt.ylabel('x2', fontsize=10)
            plt.grid()

            # Plot test samples
            plt.subplot(1, 2, 2)
            plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
            for n, m in enumerate(np.unique(y_test)):
                plt.scatter(X_test_pca[y_test == m, 0], X_test_pca[y_test == m, 1],
                            c=ListedColormap(('g', 'r', 'b'))(n),
                            s=10, cmap=cm_dark)
            plt.legend(['Fat', 'Tumor'])
            plt.xlim(x1_min, x1_max)
            plt.ylim(x2_min, x2_max)
            plt.title('Testing visualization')
            plt.suptitle(u'SVM: ' + titles[i])
            plt.xlabel('x1', fontsize=10)
            plt.ylabel('x2', fontsize=10)
            plt.grid()

            plt.tight_layout(2)
            plt.subplots_adjust(top=0.92)
            # ax.savefig('../result/SVM/SVM_TESTING/Yale/retrain_classifier/SVM_1200_noAug_area_normalization_Yale.png')
            ax.show()
            plt.pause(1)

            # ======================================================================================================================#
            # Misclassified data
            choose = input("If you wanna plot misclassified data, press 1: ")
            # choose = '1'
            if (choose == '1'):
                data_pca = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca)], ignore_index=True, axis=0)
                data_pca.columns = ['pca1', 'pca2']
                normalization_PCA = pd.concat([pd.concat(
                    [pd.concat([data, data_pca], axis=1),
                     pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], ignore_index=True, axis=0)], axis=1),
                    path],
                    axis=1)

                wrong_data_train = pd.DataFrame()
                y_train_pred = clf.predict(X_train_pca)
                print('Training set: ')
                for train in range(0, len(y_train_pred)):
                    if (y_train_pred[train] != y_train[train]):
                        print('y_train_pred: ', y_train_pred[train], ' / ', 'y_train: ', y_train[train])
                        print(y_train_pred[train] != y_train[train], train)
                        key = pd.DataFrame(X_train_pca[train]).T
                        wrong_data_train = pd.concat([wrong_data_train, key], axis=0)
                print('===================================')

                # Getting samples which is in wrong classes in testing set
                wrong_data = pd.DataFrame()
                print('Testing set: ')
                for i_pred in range(0, len(y_pred)):
                    if (y_pred[i_pred] != y_test[i_pred]):
                        print('y_pred: ', y_pred[i_pred], ' / ', 'y_test: ', y_test[i_pred])
                        print(y_pred[i_pred] != y_test[i_pred], i_pred)
                        key = pd.DataFrame(X_test_pca[i_pred]).T
                        wrong_data = pd.concat([wrong_data, key], axis=0)
                print('===================================')

                # training data
                if wrong_data_train.empty:
                    print('There is no training data points with wrong labels')
                else:
                    wrong_data_train.columns = ['pca1', 'pca2']
                    wrong_data_train.index = [index for index in range(0, len(wrong_data_train.iloc[:, 1].values))]
                    concat_normalization_wrong = pd.merge(normalization_PCA, wrong_data_train, on=['pca1', 'pca2'])

                    # Plot data using normalization
                    for mis in range(0, len(concat_normalization_wrong)):
                        fig = plt.figure(figsize=(10, 8), facecolor='w')
                        plt.plot(np.arange(paras['X_min'], paras['X_max']),
                                 concat_normalization_wrong.iloc[mis, : paras['X_max'] - paras['X_min']].values,
                                 label='True label: ' + str(
                                     int(concat_normalization_wrong.iloc[mis, -2])) + '\n' + 'File_path: ' + str(
                                     concat_normalization_wrong.iloc[mis, -1]))
                        plt.plot(np.arange(paras['X_min'], paras['X_max']), mean_value(nor_Fat), label="fat(Nor)-mean")
                        plt.plot(np.arange(paras['X_min'], paras['X_max']), mean_value(nor_Tumor),
                                 label="tumor(Nor)-mean")
                        plt.title('Training data: misclassified(Before Normalization): ' + '--' + str(mis))
                        plt.legend(bbox_to_anchor=(0, 1.02, 0, .102), loc=3,
                                   ncol=1, mode="expand", borderaxespad=0.)
                        # plt.savefig(
                        #     '../result/SVM/SVM_TESTING/MIMICS/' + str(
                        #         mis) + 'png')
                        fig.show()

                # testing set
                if wrong_data.empty:
                    print("There is no data points with wrong labels")
                else:
                    wrong_data.columns = ['pca1', 'pca2']
                    wrong_data.index = [index for index in range(0, len(wrong_data.iloc[:, 1].values))]
                    concat_normalization_wrong = pd.merge(normalization_PCA, wrong_data, on=['pca1', 'pca2'])

                    # Plot data using normalization
                    for mis in range(0, len(concat_normalization_wrong)):
                        ax = plt.figure(figsize=(10, 8), facecolor='w')
                        plt.plot(np.arange(paras['X_min'], paras['X_max']),
                                 concat_normalization_wrong.iloc[mis, : paras['X_max'] - paras['X_min']].values,
                                 label='True label: ' + str(
                                     int(concat_normalization_wrong.iloc[mis, -2])) + '\n' + 'File_path: ' + str(
                                     concat_normalization_wrong.iloc[mis, -1]))
                        plt.plot(np.arange(paras['X_min'], paras['X_max']), mean_value(nor_Fat), label="fat(Nor)-mean")
                        plt.plot(np.arange(paras['X_min'], paras['X_max']), mean_value(nor_Tumor),
                                 label='tumor(Nor)-mean')
                        plt.title('Testing data: misclassified(Normalization): ' + titles[i] + '--' + str(mis))
                        plt.legend(bbox_to_anchor=(0, 1.02, 0, .102), loc=3,
                                   ncol=1, mode="expand", borderaxespad=0.)
                        # plt.savefig(
                        #     '../result/SVM/SVM_TESTING/MIMICS/' + str(
                        #         mis) + 'png')
                        ax.show()

        plt.pause(0.1)
        if fig_interactive:
            fig_interactive.canvas.mpl_disconnect(cid_interactive)
        plt.show(block=True)  # force program to wait for plots to close before exiting
