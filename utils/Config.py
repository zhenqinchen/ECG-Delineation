import sys


import os
from enum import Enum
import pandas as pd

import time

class Config:
    RESOURCES_DIR ='resource'
    DATA_DIR =  'data'
    QTDB_RECORD_DIR = 'data/' + 'qtdb/physionet.org/files/qtdb/1.0.0/'
    LUDB_RECORD_DIR = 'data/' + 'physionet.org/files/ludb/1.0.1/'

    

    P_H = 1
    QRS_H = 3
    T_H = 2
    

    def __init__(self):
        ####signal####
        self.dataset = 'ludb'
        self.wave_len = 280
        self.fc = 250
        ####data####
        self.seed = 1
        self.data = None
        self.batch_size = 32
        self.epochs = 30
        self.lr = 1e-3
        self.kernel_size = 9
        self.conv_channels = 32
        self.train_verbose = True
        
        ####filename####
        self.refresh()

    
    
    def print(self):
        print('filename:', self.fname_data)
    
    def refresh(self):

        self.fname_data = Config.DATA_DIR+ '/' + dataset + '.pkl'
        self.fname_model = Config.RESOURCES_DIR + '/'+ dataset + '_model.h5'
        self.fname_history = Config.RESOURCES_DIR + '/l'+  dataset+ '_history.pkl'

        


    
    
    
