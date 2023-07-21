# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:41:40 2023

@author: Marios
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pymysql
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
        #return row[int(n*0.7):], label[int(n*0.7):]
        return row[int(n*0.7):int(n*0.9)], label[int(n*0.7):int(n*0.9)]
    def split_window_test(self, row, label):
        n = self.n
        return row[int(n*0.9):], label[int(n*0.9):]

class Model:
    def __init__(self, window_size, lstm_units, lstm_units_2, lstm_units_3, dense_neurons_1, dense_neurons_2, dense_neurons_3, save_path):
        self.model = Sequential()
        #self.model.add(InputLayer((input_size_1, input_size_2)))
        self.model.add(LSTM(lstm_units, return_sequences=True, input_shape=(None, window_size)))
        self.model.add(LSTM(lstm_units_2, return_sequences=True)) # 2nd lstm
        self.model.add(LSTM(lstm_units_3)) # 3rd lstm
        self.model.add(Dense(dense_neurons_1 , activation='relu'))
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
    
    def predict_sequence(self, input_data, time):
        new_predictions = []
        for i in range(time):
            input_ = np.array(input_data).reshape(1, 1, 72)
            predicted_value = self.predictions(input_data) # make prediction using input
            new_predictions.append(predicted_value)
            print(f"Predicted value: {predicted_value}")
            predicted_value = np.expand_dims(np.expand_dims(predicted_value, axis=0), axis=0) 
            
            appended_array = np.concatenate((input_, predicted_value), axis=2) # append value to the input array
            
            input_data = appended_array[:, :, 1:] # reshape the input to include the new prediction, but forget the first value
       
        return np.array(new_predictions)
    
    def plot_predictions(self, results, actual_data):
        # Create a DataFrame with train results
        train_results = pd.DataFrame(data={'Train Predictions': results * 4300, 'Actual': actual_data.flatten() * 4300})

        # Plotting the predictions and the real data
        fig, (ax1, ax2) = plt.subplots(2, sharex='col', sharey='row')
        ax1.plot(train_results['Train Predictions'])
        ax1.set_title('Predictions')
        ax2.plot(train_results['Actual'])
        ax2.set_title('Actual data')
        
        # Show the plot
        plt.show()
        return train_results


class ThresholdScanner:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.above_threshold = pd.DataFrame(columns=['Hours','Value'])
        self.current_datetime = datetime.datetime.now()

    def scan_threshold(self, column_name, threshold):
        values_found = False
        
        for index, row in self.dataframe.iterrows():
            value = row[column_name]
            if value >= threshold:
                #time = datetime.datetime.fromtimestamp(index).strftime('%Y-%m-%d %H:%M:%S')
                self.above_threshold = self.above_threshold.append({'Hours':index, 'Value':value},ignore_index=True)
                values_found = True
        if not values_found:
            self.above_threshold = self.above_threshold.append({'Hours': 0, 'Value': 0}, ignore_index=True)
        
        
        self.above_threshold.rename(columns={'Hours':'sec_from_j2000'})
        for index, row in self.above_threshold.iterrows():
            if row['Hours']> 0:
                hours = row['Hours']
    
                # Add the given number of hours to the current datetime
                target_datetime = self.current_datetime + datetime.timedelta(hours=hours)
    
                # Format the target datetime as 'dd/mm/yyyy hh:mm'
                formatted_date = target_datetime.strftime('%d/%m/%Y %H:%M')
            
                j_2000 = '01/01/2000 00:00'
                date_format = '%d/%m/%Y %H:%M'
                date1 = datetime.datetime.strptime(formatted_date, date_format)
                date2 = datetime.datetime.strptime(j_2000, date_format)
            
                sec_from_j2000 = (date1 - date2).total_seconds()
            
            else :
                sec_from_j2000 = 0
            # Update the 'Hours' column with the formatted date
            
            self.above_threshold.at[index, 'sec_from_j2000'] = sec_from_j2000
                
        return self.above_threshold
    
    def scan_results(self, wheel1_pred, wheel2_pred, wheel3_pred, wheel4_pred):
        
        new_df = pd.DataFrame()
        for i, df in enumerate([wheel1_pred, wheel2_pred, wheel3_pred, wheel4_pred]):
            # Get the first date that each wheel will reach the threshold
            
            value_col2 = df.iloc[0, 1]
            value_col3 = df.iloc[0, 2]
            
            # Append the values and dataframe name to the new dataframe
            new_df = new_df.append({ 'Wheel Speed': value_col2, 'date wrt j2000': value_col3}, ignore_index=True) #'Dataframe': f'df{i+1}'}

        new_df['Reaction Wheel'] = ['Reaction Wheel 1','Reaction Wheel 2', 'Reaction Wheel 3','Reaction Wheel 4' ]
        # Get which wheel will reach the threshold value first and when,  wrt j2000
        min_value = new_df[new_df['date wrt j2000'] > 0]['date wrt j2000'].min()
        min_value_index = new_df[new_df['date wrt j2000'] == min_value].index[0]
        wheel_to_unload = new_df.iloc[min_value_index,2]
        
        return min_value, wheel_to_unload
    
class Create_Ticket:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
    
    def connect(self):
        try: #connect to the database
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
                )
            self.cursor = self.connection.cursor()
            print('Connected to MySQL database')

        except pymysql.Error as e:
            print('Error connecting to MySQL:', e)        
        except pymysql.Error as e:
            print('Error connecting to MySQL:', e)        
    
    def get_latest_value(self):
    # retrieve the latest value from the database for the ticket id
        select_query = "SELECT MAX(ticket_code) FROM ticket"
        self.cursor.execute(select_query)
        latest_value = self.cursor.fetchone()[0]
        return latest_value
    
    def insert_ticket(self, date_unloading, saturated_wheel):
        try:
            # Get the latest entry in the database for a specific column
            latest_value = self.get_latest_value()

            # set the new ticket ID
            ticket_id= latest_value + 1
            
            insert_query = "INSERT INTO ticket (ticket_code, ticket_ticket_type_code\
                , ticket_ticket_priority_code, ticket_ticket_group_code,\
                ticket_ticket_status_code, ticket_subject, ticket_start_date,\
                ticket_due_date, ticket_desc, ticket_satellite, ticket_verified,\
                ticket_in_flag, ticket_out_flag, noc_ticket_services_code,\
                noc_ticket_issue_code, noc_ticket_operation_code)\
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data = (ticket_id, 1, 5, 1, 1, "Warning: Manual Wheel Unloading is required", date_unloading, 0, 'Wheel that will reach the limit is: '+ saturated_wheel, 3, 0, 0, 0, 0, 0, 0)
            # Execute the insertion query with the modified data
            self.cursor.execute(insert_query, data)
            self.connection.commit()
            print('Data inserted to table ticket successfully')

        except pymysql.Error as e:
            print('Error writing to MySQL:', e)


    def insert_ticket_task(self, task_description):
        try: # insert decription steps to the ticket
            ticket_id = self.get_latest_value()
            select_query = "SELECT MAX(ticket_task_code) FROM ticket_task"
            self.cursor.execute(select_query)
            task_id = self.cursor.fetchone()[0] + 1
            
            insert_query = "INSERT INTO ticket_task (ticket_task_code, ticket_task_ticket_code,\
                ticket_task_task_type_code, ticket_task_status_code, ticket_task_ticket_phase_code,\
                ticket_task_subject, ticket_task_start_date, ticket_task_due_date,\
                ticket_task_ins_date, ticket_task_desc)\
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data = (task_id, ticket_id, 1, 1, 1, "Wheel Unloading", 0, 0, 0, task_description)    
            # Execute the insertion query with the modified data
            self.cursor.execute(insert_query, data)
            self.connection.commit()
            print('Data inserted to table ticket_task successfully')
            
        except pymysql.Error as e:
            print('Error writing to MySQL:', e)

    def close_connection(self):
        if self.connection:
            self.connection.close()



data_analyzer = DataAnalyzer('C:/Users/Marios/Documents/Astronautics and Space Engineering MSc/IRP-Hellas-Sat/HS3 Data/v2_HS3_RWA_Rates_H_2022.txt')       

rw1_spd = data_analyzer.filter_by_parameter('AW1010R')
rw2_spd = data_analyzer.filter_by_parameter('AW2010R')
rw3_spd = data_analyzer.filter_by_parameter('AW3010R')
rw4_spd = data_analyzer.filter_by_parameter('AW4010R')

'''
datasplit= Datasplit(rw4_spd)
x, y = datasplit.window(72)

train_x, train_y = datasplit.split_window_train(x, y)
val_x, val_y = datasplit.split_window_validate(x, y)
test_x, test_y = datasplit.split_window_test(x, y)

# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

model_2 = Model(72, 192, 128, 32, 128, 32, 1, 'model_rw_4/')
model_2.compile_model(0.001)
model_2.training(train_x, train_y, val_x, val_y, 100)

model_2.load('model_rw_4/')

# prediction for training set
results = model_2.predictions(train_x)
train_results = model_2.plot_predictions(results, train_y)

# prediction for validation set
results_2 = model_2.predictions(val_x)
train_results_2 = model_2.plot_predictions(results_2, val_y)

# prediction for test set
results_3 = model_2.predictions(test_x)
train_results_3 = model_2.plot_predictions(results_3, test_y)
'''

datasplit1= Datasplit(rw1_spd)
x1, y1 = datasplit1.window(72)

train_x1, train_y1 = datasplit1.split_window_train(x1, y1)
val_x1, val_y1 = datasplit1.split_window_validate(x1, y1)
test_x1, test_y1 = datasplit1.split_window_test(x1, y1)

train_x1 = np.reshape(train_x1, (train_x1.shape[0], 1, train_x1.shape[1]))
val_x1 = np.reshape(val_x1, (val_x1.shape[0], 1, val_x1.shape[1]))
test_x1 = np.reshape(test_x1, (test_x1.shape[0], 1, test_x1.shape[1]))

model_1 = Model(72, 64, 192, 32, 64, 64, 1, 'model_rw_1/')

model_1.load('model_rw_1/')

# prediction for wheel 1
results1 = model_1.predictions(test_x1)
train_results1 = model_1.plot_predictions(results1, test_y1)

datasplit2= Datasplit(rw2_spd)
x2, y2 = datasplit2.window(72)

train_x2, train_y2 = datasplit2.split_window_train(x2, y2)
val_x2, val_y2 = datasplit2.split_window_validate(x2, y2)
test_x2, test_y2 = datasplit2.split_window_test(x2, y2)

train_x2 = np.reshape(train_x2, (train_x2.shape[0], 1, train_x2.shape[1]))
val_x2 = np.reshape(val_x2, (val_x2.shape[0], 1, val_x2.shape[1]))
test_x2 = np.reshape(test_x2, (test_x2.shape[0], 1, test_x2.shape[1]))

model_2 = Model(72, 128, 192, 128, 128, 64, 1, 'model_rw_2/')

model_2.load('model_rw_2/')

# prediction for wheel 2
results2 = model_2.predictions(test_x2)
train_results2 = model_2.plot_predictions(results2, test_y2)

datasplit3= Datasplit(rw3_spd)
x3, y3 = datasplit3.window(72)

train_x3, train_y3 = datasplit3.split_window_train(x3, y3)
val_x3, val_y3 = datasplit3.split_window_validate(x3, y3)
test_x3, test_y3 = datasplit3.split_window_test(x3, y3)

train_x3 = np.reshape(train_x1, (train_x3.shape[0], 1, train_x3.shape[1]))
val_x3 = np.reshape(val_x1, (val_x3.shape[0], 1, val_x3.shape[1]))
test_x3 = np.reshape(test_x1, (test_x3.shape[0], 1, test_x3.shape[1]))

model_3 = Model(72, 64, 32, 256, 32, 64, 1, 'model_rw_3/')

model_3.load('model_rw_3/')

# prediction for wheel 3
results3 = model_3.predictions(test_x3)
train_results3 = model_3.plot_predictions(results3, test_y3)

datasplit4= Datasplit(rw4_spd)
x4, y4 = datasplit4.window(72)

train_x4, train_y4 = datasplit4.split_window_train(x4, y4)
val_x4, val_y4 = datasplit4.split_window_validate(x4, y4)
test_x4, test_y4 = datasplit4.split_window_test(x4, y4)

train_x4 = np.reshape(train_x1, (train_x4.shape[0], 1, train_x4.shape[1]))
val_x4 = np.reshape(val_x1, (val_x4.shape[0], 1, val_x4.shape[1]))
test_x4 = np.reshape(test_x4, (test_x4.shape[0], 1, test_x4.shape[1]))

model_4 = Model(72, 192, 128, 32, 128, 32, 1, 'model_rw_4/')

model_4.load('model_rw_4/')

# prediction for wheel 4
results4 = model_4.predictions(test_x4)
train_results4 = model_4.plot_predictions(results4, test_y4)


# Scan the predictions and get a Dataframe saying in 
#how many hours the rpm will reach/exceed the threshold

scan = ThresholdScanner(train_results1)

above_limit = scan.scan_threshold('Train Predictions', 4000)

scan2 = ThresholdScanner(train_results2)

above_limit_2 = scan2.scan_threshold('Train Predictions', 4000)

scan3 = ThresholdScanner(train_results3)

above_limit_3 = scan3.scan_threshold('Train Predictions', 4000)

scan4 = ThresholdScanner(train_results4)

above_limit_4 = scan4.scan_threshold('Train Predictions', 4000)

date_unloading, num_wheel = scan.scan_results(above_limit, above_limit_2, above_limit_3, above_limit_4)


database=Create_Ticket('127.0.0.1', 'root', 'Marios970511', 'irp-ticket')
database.connect()
database.insert_ticket(date_unloading, num_wheel)
database.insert_ticket_task("step 1")
database.insert_ticket_task("step 2")
database.insert_ticket_task("step 3")
database.close_connection()
'''

new_df = pd.DataFrame()

# Iterate over each dataframe
for i, df in enumerate([above_limit, above_limit_2]):
    # Get the first value from each column
    
    value_col2 = df.iloc[0, 1]
    value_col3 = df.iloc[0, 2]
    
    # Append the values and dataframe name to the new dataframe
    new_df = new_df.append({ 'Wheel Speed': value_col2, 'date wrt j2000': value_col3}, ignore_index=True) #'Dataframe': f'df{i+1}'}

new_df['Reaction Wheel'] = ['Reaction Wheel 1','Reaction Wheel 2', 'Reaction Wheel 3','Reaction Wheel 4' ]
# Get the minimum value from the desired column in the new dataframe
min_value = new_df[new_df['date wrt j2000'] > 0]['date wrt j2000'].min()
min_value_index = new_df[['date wrt j2000'] > 0]['date wrt j2000'].idxmin()
wheel_to_unload = new_df.iloc[min_value_index,2]


first_input = val_x[1749,0,:]

asdf = np.array(first_input).reshape(1, 1, 72)

qwerty = model_2.predict_sequence(first_input, 802)
'''