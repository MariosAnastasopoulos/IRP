# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:34:55 2023

@author: Marios
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
#import os 
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataAnalyzer:
    def __init__(self, datafile):
        # import data
        self.df = pd.read_csv(datafile)
        # convert date and time
        self.df['End time'] = pd.to_datetime(self.df['End time'], format='%d/%m/%Y %H:%M:%S.%f')
        self.df['Start time'] = pd.to_datetime(self.df['Start time'], format='%d/%m/%Y %H:%M:%S.%f')
        # set datetime as index
        self.df = self.df.set_index('End time')

    def filter_by_parameter(self, telemetry_code):
        self.df = self.df[['Parameter','Mean']]
        # filter data by telemetry code
        return self.df[self.df['Parameter'] == telemetry_code]

class Datasplit:
    def __init__(self, dataset):
        self.df = dataset.drop(columns=["Parameter"])
        self.n = len(dataset)
        
    def window(self, window_size):
        data = self.df.to_numpy()
        data = np.absolute(data)
        x = []
        y = []
        for i in range(self.n-window_size):
            row = [a for a in data[i:i+window_size]]  # [1 2 3 4 5] [6]
            x.append(row)                             # [2 3 4 5 6] [7]
            label = data[i+window_size]               # ...
            y.append(label)
        return np.array(x) , np.array(y)
    def split_window_train(self, row, label):
        n = self.n
        return row[0:int(n*0.7)], label[0:int(n*0.7)]
    def split_window_validate(self, row, label):
        n = self.n
        return row[int(n*0.7):int(n*0.9)], label[int(n*0.7):int(n*0.9)]
    def split_window_test(self, row, label):
        n = self.n
        return row[int(n*0.9):], label[int(n*0.9):]
 

class Model:
    def __init__(self, input_size_1, input_size_2, lstm_units, dense_neurons_1, dense_neurons_2, save_path):
        self.model = Sequential()
        self.model.add(InputLayer((input_size_1, input_size_2)))
        self.model.add(LSTM(lstm_units))
        self.model.add(Dense(dense_neurons_1 , activation='relu'))
        self.model.add(Dense(dense_neurons_2 , activation='linear'))
        self.cp = ModelCheckpoint(save_path , save_best_only=True)
        
    def compile_model(self, learning_rate):
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate= learning_rate), metrics=[RootMeanSquaredError()])
        
    def training(self, train_set_x, train_set_y, val_set_x, val_set_y, epoch):
        #self.model.fit(train_set_x , train_set_y , validation_data=(val_set_x , val_set_y ), epochs= epoch, callbacks=[self.cp])
        history = self.model.fit(train_set_x , train_set_y , validation_data=(val_set_x , val_set_y ), epochs= epoch, callbacks=[self.cp])
        # Plot the training and validation losses
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        #Plot the training and validation metrics
        plt.subplot(1, 2, 2)
        plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
        plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()

        plt.tight_layout()
        plt.show()
                 
    def load(self, save_path):
        self.model = load_model(save_path)
        
    def predictions(self, dataset):
        return self.model.predict(dataset).flatten()

data_analyzer = DataAnalyzer('C:/Users/Marios/Documents/Astronautics and Space Engineering MSc/IRP-Hellas-Sat/HS3 Data/v2_HS3_RWA_Rates_H_2022.txt')       
#rw1_spd = data_analyzer.filter_by_parameter('AW1010R')
#rw2_spd = data_analyzer.filter_by_parameter('AW2010R')
#rw3_spd = data_analyzer.filter_by_parameter('AW3010R')
#rw4_spd = data_analyzer.filter_by_parameter('AW4010R')
#sc_mntm_x = data_analyzer.filter_by_parameter('AW0004R')
sc_mntm_y = data_analyzer.filter_by_parameter('AW0005R')
#sc_mntm_z = data_analyzer.filter_by_parameter('AW0006R')
#sc_mntm_tot = data_analyzer.filter_by_parameter('HA0677D')
"""
rw_sum1 = rw1_spd.append(rw2_spd)
rw_sum2 = rw_sum1.append(rw3_spd)
rw_total = rw_sum2.append(rw4_spd)
"""
datasplit= Datasplit(sc_mntm_y)
x, y = datasplit.window(24)

train_x, train_y = datasplit.split_window_train(x, y)
val_x, val_y = datasplit.split_window_validate(x, y)
test_x, test_y = datasplit.split_window_test(x, y)

  
model_mntm_y = Model(24, 1, 128,192, 1, 'model_y_mntm/')
#model_mntm_y.compile_model(0.001)
#model_mntm_y.training(train_x, train_y, val_x, val_y, 20)

model_mntm_y.load('model_y_mntm/')
results = model_mntm_y.predictions(test_x)

train_results = pd.DataFrame(data={'Train Predictions':results,'Actual':test_y.flatten()})

# plotting the predictions and the real data
fig, (ax1, ax2) = plt.subplots(2, sharex='col', sharey='row')
ax1.plot(train_results['Train Predictions'])
ax1.set_title('Predictions')
ax2.plot(train_results['Actual'])
ax2.set_title('Actual data')

