# -*- coding: utf-8 -*-
"""
@author: Marios
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

#from tensorflow import keras
#from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters


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
    def __init__(self, window_size, save_path, epoch):
        self.model = None
        self.window_size = window_size
        self.save_path = save_path
        self.epoch = epoch
        self.cp = ModelCheckpoint(save_path, save_best_only=True)

    def build_model(self, hp):
        model = Sequential()
        #model.add(InputLayer(input_shape=(self.input_size_1, self.input_size_2)))
        
        # Add LSTM layer with variable number of units
        hp_units = hp.Choice('lstm_units', values=[32,64,128,192,256])
        #hp_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
        model.add(LSTM(units=hp_units , return_sequences=True, input_shape=(None, self.window_size)))
        
        # Add 2nd LSTM layer with variable number of units
        hp_units_2 = hp.Choice('lstm_units_2', values=[32,64,128,192,256])
        #hp_neurons = hp.Int('dense_neurons_1', min_value=64, max_value=256, step=64)
        #hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
        model.add(LSTM(units=hp_units_2, return_sequences=True))
        
        # Add 3rd LSTM layer with variable number of units
        hp_units_3 = hp.Choice('lstm_units_3', values=[32,64,128,192,256])
        #hp_neurons = hp.Int('dense_neurons_1', min_value=64, max_value=256, step=64)
        #hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
        model.add(LSTM(units=hp_units_3))       
        
        # Add dense layer with variable number of neurons and activators
        hp_neurons_2 = hp.Choice('dense_neurons_2', values=[32,64,128,192,256])
        #hp_neurons = hp.Int('dense_neurons_1', min_value=64, max_value=256, step=64)
        hp_activation_2 = hp.Choice('activation', values=['relu', 'tanh'])
        model.add(Dense(units=hp_neurons_2, activation=hp_activation_2))
        
        model.add(Dense(units=1, activation='linear'))
        
        # Variable learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])
        
        # Compile model 
        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss=MeanSquaredError(),
            metrics=[RootMeanSquaredError()],
        )

        return model
    
    """ #Random Search
    def hyperparameter_search(self, train_set_x, train_set_y, val_set_x, val_set_y): 
        tuner = RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=3,  # Adjust the number of trials as desired
            executions_per_trial=1,
            directory='hyperparameter_search',
            project_name='model_search',
        )

        tuner.search(train_set_x, train_set_y, validation_data=(val_set_x, val_set_y), epochs=self.epoch, callbacks=[self.cp])

        best_model = tuner.get_best_models(num_models=1)[0]
        self.model = best_model
    """ 
    
    # Hyperband Search
    def hyperparameter_search(self, train_set_x, train_set_y, val_set_x, val_set_y):
        tuner = Hyperband(
            self.build_model,
            objective='val_loss',
            max_epochs= self.epoch,  # Set the maximum number of epochs for each trial
            factor=4,  # Adjust the factor 
            directory='hyperparameter_search',
            project_name='model_search'
        )

        tuner.search(train_set_x, train_set_y, validation_data=(val_set_x, val_set_y))

        best_model = tuner.get_best_models(num_models=1)[0]
        self.model = best_model

    def training(self, train_set_x, train_set_y, val_set_x, val_set_y):
        if self.model is None:
            self.hyperparameter_search(train_set_x, train_set_y, val_set_x, val_set_y)

        self.model.fit(train_set_x, train_set_y, validation_data=(val_set_x, val_set_y), epochs=self.epoch, callbacks=[self.cp])
        history = self.model.history

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def print_summary(self):
        self.model.summary()
        learning_rate = self.model.optimizer.lr.numpy()
        print(f"Learning Rate: {learning_rate}")
        #activation = self.model.Dense.activation.__name__
        #print(f"Layer: Dense, Activation: {activation}")
        for layer in self.model.layers:
                if isinstance(layer, Dense):
                    activation = layer.activation.__name__
                    print(f"Layer: {layer.name}, Activation: {activation}")
        
#Remove the tuner state, so that a fresh hp search can be perfomed
tuner_state_file = "hyperparameter_search/model_search/tuner0.json"
if os.path.exists(tuner_state_file):
    os.remove(tuner_state_file)

data_analyzer = DataAnalyzer('C:/Users/Marios/Documents/Astronautics and Space Engineering MSc/IRP-Hellas-Sat/HS3 Data/v2_HS3_RWA_Rates_H_2022.txt')       

#sc_mntm_tot = data_analyzer.filter_by_parameter('AW0006R')
#rw1_spd = data_analyzer.filter_by_parameter('AW1010R')
rw2_spd = data_analyzer.filter_by_parameter('AW2010R')
#rw3_spd = data_analyzer.filter_by_parameter('AW3010R')
#rw4_spd = data_analyzer.filter_by_parameter('AW4010R')
#sc_mntm_x = data_analyzer.filter_by_parameter('AW0004R')
#sc_mntm_y = data_analyzer.filter_by_parameter('AW0005R')
#sc_mntm_z = data_analyzer.filter_by_parameter('AW0006R')
#
"""
rw_sum1 = rw1_spd.append(rw2_spd)
rw_sum2 = rw_sum1.append(rw3_spd)
rw_total = rw_sum2.append(rw4_spd)
"""

datasplit = Datasplit(rw2_spd)

#datasplit= Datasplit(sc_mntm_tot)
x, y = datasplit.window(72)

train_x, train_y = datasplit.split_window_train(x, y)
val_x, val_y = datasplit.split_window_validate(x, y)
test_x, test_y = datasplit.split_window_test(x, y)

# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))


model_3= Model(72, 'model_rw_4/', 30)

model_3.hyperparameter_search(train_x, train_y, val_x, val_y)

model_3.print_summary()


