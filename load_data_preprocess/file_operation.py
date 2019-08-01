#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''@Author : Yang Li
# @Time : 7/11/2019 9:44 AM
# @File : load_data.py
# @Description :
                Containing all file operations
'''
import os
import re


def get_file_path(dir):
    ''' Getting all paths to the file

    :param dir: String containing the path to the most senior file
    :return: Array containing all paths to the file
    '''
    files_ = []
    file_names = os.listdir(dir)
    file_names.sort()
    for i in range(0, len(file_names)):
        path = os.path.join(dir, file_names[i])
        if os.path.isdir(path):
            files_.extend(get_file_path(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_


def get_file_name(file_path):
    ''' Extracting file name from a path

    :param file_path: String containing a path to the  file
    :return: String containing the name to the file
    '''
    file_name = os.path.basename(file_path)
    # The other way: split character is decided by directory. '/' or '\\'
    # file_name = file_path.split("\\")[-1]
    return file_name


def extract_cv(files_):
    ''' Selecting all files with suffix name _cv

    :param files_: Array containing the paths to the files
    :return: Array containing all files with suffix name _cv
    '''
    _cv = []
    re1 = '.*?'  # Non-greedy match on filler
    re2 = '(cv\\.csv)'  # Fully Qualified Domain Name 1
    for i in range(0, len(files_)):
        rg = re.compile(re1 + re2)
        m = rg.search(files_[i])
        if m != None:
            linebits = m.group()
            _cv.append(linebits)
    return _cv


def regular_expression(filename, tag):
    '''

    :param filename: String
    :param tag: Int
    :return:
    '''
    reg1 = re.compile(r'^(?P<name>[^ ]*)_(?P<num>[^ ]*)_(?P<rest>[^ ]*)_')
    reg2 = re.compile(r'^(?P<label>[^ ]*)__')
    if tag == 1:
        reg_match = reg1.match(filename).groupdict()
        return reg_match['num']
    if tag == 2:
        reg_match = reg2.match(filename).groupdict()
        return reg_match['label']
    if tag == 3:
        required_element = []
        reg = re.compile(
            r'^(?P<name>[^ ]*)__Raman(?P<Raman>[^ ]*)__(?P<version>[^ ]*)__(?P<ccw>[^ ]*)__(?P<precessed>[^ ]*)__(?P<rest>[^ ]*)')
        reg_match = reg.match(filename).groupdict()
        required_element.append(reg_match['name'])
        required_element.append(reg_match['version'])
        return required_element
