import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras



def init_gpus():
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            #Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            #Memory growth must be set before GPUs have been initialized
            print(e)


def read_dat_file(filename):
    """
    Read .dat file, discards some row headers and returns appropriate values.

    Parameters
    ----------
    filename : string with path and filename do .dat file

    Returns
    -------
    df : pandas.DataFrame
        A pandas dataframe contatining the data.
    """
    df = pd.read_csv(filename, skiprows=3)
    df_aux = pd.read_csv(filename, header=1)
    df.columns = df_aux.columns

    cols_to_drop = ['RECORD', 'Excedente_Avg', 'Compra_Avg']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop([col], axis=1)

    for column in df.columns:
        if column != "TIMESTAMP":
            df[column] = df[column].astype('float')
    # Drop column 'RECORD' (if present) because from june 2019 is is no longer used
    return df


def get_list_of_files(folder):
    """
    Return a list of *.dat files inside the subfolders of folder 'folder'.

    Parameters
    ----------
    folder : string with path to root folder

    Returns
    -------
    lst : list
        A list containing all *.dat file strings
    """
    lst = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            complete_filename = os.path.join(root, name)
            # print(complete_filename)
            lst.append(complete_filename)
        for name in dirs:
            complete_filename = os.path.join(root, name)
            # print(complete_filename)
            lst.append(complete_filename)

    lst.sort()
    return [x for x in lst if '.dat' in x]

#Preprocessing Data
def split_sequence(sequence, n_steps_in, n_steps_out,input_labels,output_labels):
    """
    Splits a univariate sequence into samples
    Parameters
    ----------
    data : Series
        Series containing the data sequences you want to divide into samples of inputs and outputs for the model prediction.
    n_steps_in : Integer
        Number of past data which will be used in a single input samples.
        How many minutes back we use to predict the next n_steps_out minutes.
    n_steps_out : Integer
        Number of future data which will represent a single output samples.
        How many minutes we will want to predict.
    input_labels : List
        List of String containnning the labels of the input of the model 
    output_labels : List
        List of String containnning the labels of the output of the model 
    Returns
    -------
    X,y : Numpy Arrays
        X represents the inputs samples and y the outputs samples.

    """
    X, y = list(), list()
    input_sequence = sequence[input_labels]
    output_sequence = sequence[output_labels]
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = input_sequence[i:end_ix], output_sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
    	
def generate_train_test_valid(folder, input_labels, output_labels, n_steps_in, n_steps_out):
    
    """
    Divides the complete dataset into training and test sets.
    
    Parameters
    ----------
    folder : String
        String containing the  path to the root folder where all the data is stored.
    input_labels : List
        List of String containnning the labels of the input of the model.
    output_labels : List
        List of String containnning the labels of the output of the model.
    n_steps_in : Integer
        Number of past data which will be used in a single input samples.
        How many minutes back we use to predict the next n_steps_out minutes.    
    n_steps_out : Integer
        Number of fu

     
    """
    #Importing the available data from 2020.
    lst = get_list_of_files(folder)
    dfs = (read_dat_file(f) for f in lst)
    df_complete = pd.concat(dfs, ignore_index=True)
        
    #Removing not a number from the dataset
    df_label = df_complete[input_labels]
    df_label = df_label.dropna()
    df_label = df_label.reset_index(drop=True)
    #Negative numbers turn into 0 for each label
    for i in input_labels:
        indexAux = df_label[(df_label[i] < 0)].index
        df_label[i][indexAux] = 0.0
    
    #Splitting the data into training and test data.
    trainingData, testData =  train_test_split(df_label,test_size = 0.1, shuffle=False)
    trainingData = trainingData.reset_index(drop=True)
    testData = testData.reset_index(drop=True)
    #Splitting the test data into input and output.
    inputData, outputData = split_sequence(testData, n_steps_in, n_steps_out, input_labels, output_labels)
    # Flatten the shape of the input samples. 
    #MLPs require that the shape of the input portion of each sample is a vector. 
    #With a multivariate input, we will have multiple vectors, one for each time step.
    if n_steps_in > 1:    
        inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
    if n_steps_out > 1:
        outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])
    pd.DataFrame(inputData).to_csv((r'./db/data/testInputData.csv'), index = False)
    pd.DataFrame(outputData).to_csv((r'./db/data/testOutputData.csv'), index = False)

    #Splitting the training data into input and output.
    inputData, outputData = split_sequence(trainingData, n_steps_in, n_steps_out, input_labels, output_labels)
    if n_steps_in > 1:    
        inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
    if n_steps_out > 1:
        outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])
    pd.DataFrame(inputData).to_csv((r'./db/data/trainingInputData.csv'), index = False)
    pd.DataFrame(outputData).to_csv((r'./db/data/trainingOutputData.csv'), index = False)

def preprocess_time(df):    
    timestamps = [ts.split('+')[0] for ts in  df['TIMESTAMP']]
    #timestamps_minute = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').minute) for t in timestamps])
    timestamps_hour = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour) for t in timestamps])
    #timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day) for t in timestamps])
    timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').month) for t in timestamps])

    minutes_in_hour = 60
    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    #df['sin_minute'] = np.sin(2*np.pi*timestamps_minute/minutes_in_hour)
    #df['cos_minute'] = np.cos(2*np.pi*timestamps_minute/minutes_in_hour)
    df['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
    df['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
    #df['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
    #df['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
    df['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
    df['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)

    return df


def generate_train_test_valid_time(folder, input_labels, output_labels, n_steps_in, n_steps_out):
    lst = get_list_of_files(folder)
    dfs = (read_dat_file(f) for f in lst)
    df_complete = pd.concat(dfs, ignore_index=True)
        
    df_label = df_complete[input_labels+["TIMESTAMP"]]
    df_label = df_label.dropna()
    df_label = df_label.reset_index(drop=True)

    for i in input_labels:
        indexAux = df_label[(df_label[i] < 0)].index
        df_label[i][indexAux] = 0.0
    
    df_label = preprocess_time(df_label)
    #input_labels = input_labels + ['sin_minute', 'cos_minute', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
    input_labels = input_labels + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']


    trainingData, testData =  train_test_split(df_label,test_size = 0.1, shuffle=False)
    trainingData = trainingData.reset_index(drop=True)
    testData = testData.reset_index(drop=True)


    inputData, outputData = split_sequence(testData, n_steps_in, n_steps_out, input_labels, output_labels)
    if n_steps_in > 1:    
        inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
    if n_steps_out > 1:
        outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])
    pd.DataFrame(inputData).to_csv((r'./db/data/testInputDataTime.csv'), index = False)
    pd.DataFrame(outputData).to_csv((r'./db/data/testOutputDataTime.csv'), index = False)

    inputData, outputData = split_sequence(trainingData, n_steps_in, n_steps_out, input_labels, output_labels)
    if n_steps_in > 1:    
        inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
    if n_steps_out > 1:
        outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])
    pd.DataFrame(inputData).to_csv((r'./db/data/trainingInputDataTime.csv'), index = False)
    pd.DataFrame(outputData).to_csv((r'./db/data/trainingOutputDataTime.csv'), index = False)





#Evaluation functions
def mae_multi(y_true, y_pred):
    """
    Mean Absolute Square (RMS) error between real Avarage PV Power and the predictions    
    ----------
    y_true : List
        List containing the real outputs.
    y_pred : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    mae : EagerTensor
    """
    #axis=0 to calculate for eache sample.
    return keras.backend.mean(keras.backend.abs(tf.math.subtract(y_true, y_pred)), axis =0)
def root_mean_square_error(y_true, y_pred):
    """
    Root Mean Square (RMS) error between real Avarage PV Power and the predictions    
    ----------
    y_true : List
        List containing the real outputs.
    y_pred : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    rms : EagerTensor
    """
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(tf.math.subtract(y_true, y_pred)), axis =0))
def standard_deviation_error(y_true, y_pred):
    """
    Standart Deviation of the error between real Avarage PV Power and the prediction    
    ----------
    y_true : List
        List containing the real outputs.
    y_pred : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    rms : EagerTensor
    """
    return keras.backend.std(tf.math.subtract(y_true, y_pred), axis =0)

def mae_rmse_stddev_evaluation(y, y_hat):
    """
    Evaluetes the prediction of a model in 3 aspects: 
        1- Mean Absolute Error (MAE) between real Avarage PV Power and the predictions
        2- Root Mean Square (RMS) error between real Avarage PV Power and the predictions 
        3- Standart Deviation of the error between real Avarage PV Power and the prediction
    ----------
    y : List
        List containing the real outputs.
    y_hat : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    mae : float
        Mean Absolute Error.
    rmse : float
        Root Mean Square
    stddev : float
        Standart Deviation
    """
    
    mae = mae_multi(y, y_hat)
    rmse = root_mean_square_error(y, y_hat)
    stddev = standard_deviation_error(y, y_hat)
    
    return mae.numpy(), rmse.numpy(), stddev.numpy()

#Baseline for univariate prediction
def generate_baseline(folder, label, time_shift, day_only):
    
    """
   Generates a baseline, predicting a label using the value of itself time_shift minutes before and
   displays the results in form of: Mean Absolute Error (MAE), 
   Root Mean Square Error (RMS) and Standart Deviation.
   
    
    Parameters
    ----------
    folder : String
        String containing the  path to the root folder where all the data is stored.

    label : String
        String containing the  name of the column of the dataset you want to use in your model.
    time_shift : Integer
        How many minutes we shift back for our prediction.   
    day_only : Bool
        True to remove the night values, False to leave them.      
    Returns
    -------
    mae : float
        Mean Absolute Error.
    rms : float
        Root Mean Square
    stddev : float
        Standart Deviation
     
    """
    #Importing the available data from 2020.
    lst = get_list_of_files(folder)
    dfs = (read_dat_file(f) for f in lst)
    df_complete = pd.concat(dfs, ignore_index=True)
    
    #Data preparation
    #Leaving or removing night time data
    if day_only == True:
        remove_index = []
        for i in range(len(df_complete['TIMESTAMP'])):
            hour_min = int(df_complete['TIMESTAMP'][i][11:13]+df_complete['TIMESTAMP'][i][14:16])
            if (hour_min  < 600) or (hour_min > 1800):
                remove_index.append(i)
        df_complete.drop(remove_index,inplace=True)
        df_complete.reset_index(drop=True,inplace=True)
    #Removing not a number from the dataset
    df_label = df_complete[label]
    df_label = df_label.dropna()
    df_label = df_label.reset_index(drop=True)
    #Negative numbers turn into 0 for each label
    indexAux = df_label[(df_label < 0)].index
    df_label[indexAux] = 0.0
    
    #List with the real and predicted (shifted 24h hours back) valeus of radiation
    y = []
    y_hat = []
    for i in range(time_shift, np.size(df_label)-1):
        y.append(df_label[i])
        y_hat.append(df_label[i-time_shift])
    
    mae, rms, stddev = mae_rmse_stddev_evaluation(y, y_hat)
    
    print("AVARAGE " +label+ " baseline prediction:")
    print("Mean Absolute Error (MAE): {:.3f}".format(mae))
    print("Root Mean Square Error (RMS): {:.3f}".format(rms))
    print("Standart Deviation: {:.3f}".format(stddev))
    
    return mae, rms, stddev