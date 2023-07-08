# -*- coding: utf-8 -*-
"""
@author: Marios
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

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
        data = data/4300 # normalize to the maximum allowed limit
        x = []
        y = []
        
        for i in range(len(data)-window_size-1):
            #print(i)
            window = data[i:(i+window_size), 0]
            x.append(window)
            y.append(data[i+window_size, 0])
    
        return np.array(x),np.array(y)

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
    def __init__(self, window_size, lstm_units, lstm_units_2, lstm_units_3, dense_neurons_2, dense_neurons_3, save_path):
        self.model = Sequential()
        #self.model.add(InputLayer((input_size_1, input_size_2)))
        self.model.add(LSTM(lstm_units, return_sequences=True, input_shape=(None, window_size)))
        self.model.add(LSTM(lstm_units_2, return_sequences=True)) # 2nd lstm
        self.model.add(LSTM(lstm_units_3)) # 3rd lstm
        #self.model.add(Dense(dense_neurons_1 , activation='relu'))
        self.model.add(Dense(dense_neurons_2 , activation='relu'))
        self.model.add(Dense(dense_neurons_3 , activation='linear'))
        self.cp = ModelCheckpoint(save_path , save_best_only=True)
        
    def compile_model(self, learning_rate):
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate= learning_rate), metrics=[RootMeanSquaredError()])
        
    def training(self, train_set_x, train_set_y, val_set_x, val_set_y, epoch):
        #self.model.fit(train_set_x , train_set_y , validation_data=(val_set_x , val_set_y ), epochs= epoch, callbacks=[self.cp])
        history = self.model.fit(train_set_x , train_set_y , validation_data=(val_set_x , val_set_y ), epochs= epoch, verbose=2 ,callbacks=[self.cp])
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


class ThresholdScanner:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.above_threshold = []
    
    def scan_threshold(self, column_name, threshold):
        for index, row in self.dataframe.iterrows():
            value = row[column_name]
            if value >= threshold:
                #time = datetime.datetime.fromtimestamp(index).strftime('%Y-%m-%d %H:%M:%S')
                self.above_threshold.append((index,value))
        return self.above_threshold



data_analyzer = DataAnalyzer('C:/Users/Marios/Documents/Astronautics and Space Engineering MSc/IRP-Hellas-Sat/HS3 Data/v2_HS3_RWA_Rates_H_2022.txt')       

#rw1_spd = data_analyzer.filter_by_parameter('AW1010R')
#rw2_spd = data_analyzer.filter_by_parameter('AW2010R')
rw3_spd = data_analyzer.filter_by_parameter('AW3010R')
#rw4_spd = data_analyzer.filter_by_parameter('AW4010R')
#sc_mntm_x = data_analyzer.filter_by_parameter('AW0004R')
#sc_mntm_y = data_analyzer.filter_by_parameter('AW0005R')
#sc_mntm_z = data_analyzer.filter_by_parameter('AW0006R')
#sc_mntm_tot = data_analyzer.filter_by_parameter('HA0677D')

datasplit= Datasplit(rw3_spd)
x, y = datasplit.window(72)

train_x, train_y = datasplit.split_window_train(x, y)
val_x, val_y = datasplit.split_window_validate(x, y)
test_x, test_y = datasplit.split_window_test(x, y)

# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

model_2 = Model(72, 256, 192, 64, 64, 1, 'model_rw_3/')
#model_2.compile_model(0.001)
#model_2.training(train_x, train_y, val_x, val_y, 50)

model_2.load('model_rw_3/')
results = model_2.predictions(train_x)
results_2 = model_2.predictions(val_x)
results_3 = model_2.predictions(test_x)

train_results = pd.DataFrame(data={'Train Predictions':results*4300,'Actual':train_y.flatten()*4300})

# plotting the predictions and the real data
fig, (ax1, ax2) = plt.subplots(2, sharex='col', sharey='row')
ax1.plot(train_results['Train Predictions'])
ax1.set_title('Predictions')
ax2.plot(train_results['Actual'])
ax2.set_title('Actual data')


train_results_2 = pd.DataFrame(data={'Train Predictions':results_2*4300,'Actual':val_y.flatten()*4300})

# plotting the predictions and the real data
fig, (ax1, ax2) = plt.subplots(2, sharex='col', sharey='row')
ax1.plot(train_results_2['Train Predictions'])
ax1.set_title('Predictions')
ax2.plot(train_results_2['Actual'])
ax2.set_title('Actual data')


train_results_3 = pd.DataFrame(data={'Train Predictions':results_3*4300,'Actual':test_y.flatten()*4300})

# plotting the predictions and the real data
fig, (ax1, ax2) = plt.subplots(2, sharex='col', sharey='row')
ax1.plot(train_results_3['Train Predictions'])
ax1.set_title('Predictions')
ax2.plot(train_results_3['Actual'])
ax2.set_title('Actual data')

scan = ThresholdScanner(train_results_3)

above_limit = scan.scan_threshold('Train Predictions', 4000)



