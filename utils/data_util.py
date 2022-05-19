
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import wfdb
from random import shuffle
import random
import sys
from .Config import Config
from . import file_util as fu


def get_ids(folder_path=None):
    filename = folder_path + '/RECORDS'
    fp = open(filename)
    ID = fp.read()
    ID = np.asanyarray(ID.split('\n'))[:-1]
    fp.close()
    return np.array(ID)

def denoise(signal, fs = 250, lowcut=0.5, highcut=40., order=3):
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    if low>0 and high>0:
        b, a = butter(order, [low, high], btype="bandpass")
    elif high>0:
        b, a = butter(order, high, btype="lowpass")
    else:
        b, a = butter(order, low, btype="highpass")
    filtedSignal = filtfilt(b, a, signal)

    return np.array(filtedSignal)

def down_sample(signal, times,method = 'interval'):
    origin_L = len(signal)
    now_L = origin_L//times
    if method == 'mean':
        signal_new = np.zeros((now_L))
        for i in range(now_L):
            start = i*times
            end = i*times + times
            if end > origin_L - 1:
                end = origin_L - 1
            signal_new[i] = np.sum(signal[start : end])/times
    elif method == 'interval':
        signal_new = signal[range(0,origin_L,times)]
    return signal_new


def get_record_list(dataset = 'qtdb'):
    if dataset == 'qtdb':
        record_list = get_ids(Config.QTDB_RECORD_DIR)
    elif dataset == 'ludb':
        record_list = get_ids(Config.LUDB_RECORD_DIR)
    
    return record_list

    
def __get_data__(data, labels, idxes, config):
    P_H = Config.P_H
    QRS_H = Config.QRS_H
    T_H = Config.T_H
    x = []
    y = []
    y_points = []
    y_names = []
    
    random.seed(config.seed)
    SHIFT = 40
    
    for idx in idxes:
        name_idx = 1
        for i in range(len(data[idx])):

            signal = data[idx][i]
            tmp = np.array(labels[idx][i])
            if config.set_zero:
                signal[:,:tmp[0]] = 0
                signal[:,tmp[5]:] = 0
          
            x.append((signal-np.mean(signal))/np.std(signal))

            label = np.zeros((config.wave_len))
            if tmp[0] != -1:
                label[tmp[0]:tmp[1]] = P_H
            if tmp[2] != -1:
                label[tmp[2]:tmp[3]] = QRS_H
            if tmp[4] != -1:
                label[tmp[4]:tmp[5]] = T_H
           
            y.append(label)
            y_names.append(idx + '_' +str(name_idx))
            name_idx += 1
            y_points.append(tmp)
    
    x = np.array(x,dtype = np.float32)
    y = np.array(y,dtype = np.float32)
    y_points = np.array(y_points, dtype = np.int32)
    y_names = np.array(y_names)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    
    return x,y, y_points, y_names
