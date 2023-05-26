# -*- coding: utf-8 -*-
"""
@author: Marios
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

class DataAnalyzer:
    def __init__(self, datafile):
        # import data
        self.df = pd.read_csv(datafile)
        # convert date and time
        self.df['End time'] = pd.to_datetime(self.df['End time'], format='%d/%m/%Y %H:%M:%S.%f')
        self.df['Start time'] = pd.to_datetime(self.df['Start time'], format='%d/%m/%Y %H:%M:%S.%f')

    def filter_by_parameter(self, telemetry_code):
        # filter data by telemetry code
        return self.df[self.df['Parameter'] == telemetry_code]

    def plot_sc_momentum(self, sc_momentum):
        # create plot for momentum 
        fig, ax = plt.subplots()
        ax.plot(sc_momentum['End time'], sc_momentum['Mean'], label=sc_momentum['Parameter'].iloc[0])
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean [Nms]')
        ax.set_title(f'{sc_momentum["Parameter"].iloc[0]}')
        plt.show()
        
    def plot_rw_speed(self, rw_speed):
        # create plot for wheel speed
        fig, ax = plt.subplots()
        ax.plot(rw_speed['End time'], rw_speed['Mean'], label=rw_speed['Parameter'].iloc[0])
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean [rpm]')
        ax.set_title(f'{rw_speed["Parameter"].iloc[0]}')
        plt.show()
        
    def frequency_scan(self, data):
        X = fft(data['Mean'])
        N = len(X)
        n = np.arange(N)
        # get the sampling rate
        sr = 1 / (60*60)
        T = N/sr
        freq = n/T
        # Get the one-sided specturm
        n_oneside = N //2
        # get the one side frequency
        f_oneside = freq[:n_oneside]
        # convert freq to days
        t_d = 1/f_oneside/(60*60*24)
        plt.figure(figsize = (12, 6))
        plt.plot(t_d, np.abs(X[:n_oneside]/n_oneside), 'b')
        plt.xticks([1, 7, 14, 30, 60, 365])
        plt.xlabel('Freq (days)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        plt.title(data["Parameter"].iloc[0])
        plt.show()


data_analyzer = DataAnalyzer('C:/Users/Marios/Documents/Astronautics and Space Engineering MSc/IRP-Hellas-Sat/HS3 Data/v2_HS3_RWA_Rates_H_2022.txt')

# create seperate datasets for each parameter
sc_mntm_x = data_analyzer.filter_by_parameter('AW0004R')
sc_mntm_y = data_analyzer.filter_by_parameter('AW0005R')
sc_mntm_z = data_analyzer.filter_by_parameter('AW0006R')
sc_mntm_tot = data_analyzer.filter_by_parameter('HA0677D')
rw1_spd = data_analyzer.filter_by_parameter('AW1010R')
rw2_spd = data_analyzer.filter_by_parameter('AW2010R')
rw3_spd = data_analyzer.filter_by_parameter('AW3010R')
rw4_spd = data_analyzer.filter_by_parameter('AW4010R')

# plot each dataset
data_analyzer.plot_sc_momentum(sc_mntm_x)
data_analyzer.plot_sc_momentum(sc_mntm_y)
data_analyzer.plot_sc_momentum(sc_mntm_z)
data_analyzer.plot_sc_momentum(sc_mntm_tot)
data_analyzer.plot_rw_speed(rw1_spd)
data_analyzer.plot_rw_speed(rw2_spd)
data_analyzer.plot_rw_speed(rw3_spd)
data_analyzer.plot_rw_speed(rw4_spd)
    
# frequency scan
data_analyzer.frequency_scan(rw1_spd)
data_analyzer.frequency_scan(rw2_spd)
data_analyzer.frequency_scan(rw3_spd)
data_analyzer.frequency_scan(rw4_spd)
data_analyzer.frequency_scan(sc_mntm_x)
data_analyzer.frequency_scan(sc_mntm_y)
data_analyzer.frequency_scan(sc_mntm_z)
data_analyzer.frequency_scan(sc_mntm_tot)  
    
  
    
  
    
  
    
  
    
  