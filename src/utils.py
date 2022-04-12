#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:00:21 2019

@author: tvieira
"""

#%% Import packages
import os
import json
import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import preprocessing
from sklearn.model_selection import KFold ,train_test_split
import joblib

from tensorflow.keras import backend, metrics, optimizers
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K



print('\nImporting packages')
print(json.__name__ + ' version ' + json.__version__)
print(np.__name__ + ' version ' + np.__version__)
print(pd.__name__ + ' version ' + pd.__version__)
print(sns.__name__ + ' version ' + sns.__version__)
# print(skl.__name__ + ' version ' + skl.__version__)
print(tf.__name__ + ' version ' + tf.__version__)
print('\n')

print("Is TensorFlow built with CUDA support? " + str(tf.test.is_built_with_cuda()))

#%% Plot properties
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)

boxprops = dict(linestyle='-',
                linewidth=3,
                color='red')
flierprops = dict(marker='o',
                  markerfacecolor='green',
                  markersize=12,
                  linestyle='none')
medianprops = dict(linestyle='-',
                   linewidth=4,
                   color='black')
meanpointprops = dict(marker='D',
                      markeredgecolor='green',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='-',
                     linewidth=4, color='blue')
whiskerprops = dict(linestyle='-',
                    linewidth=3,
                    color='red')

#%% Define important variables
titles = ['Temperature',            # 0
          'Atmospheric pressure',   # 1
          'Humidity',               # 2
          'Irradiance',             # 3
          'Velocity',               # 4
          'Panel temp.',            # 5
          'Voltage',                # 6
          'Current']                # 7

feats = ['TEMPERATURA', 
         'PRESSAO', 
         'UMIDADE', 
         'IRRADIANCIA',
         'VELOCIDADE',
         'TEMP_PAINEL',
         'TENSAO',
         'CORRENTE']

units = ['Celsius', 
         'hPa', 
         '$\%$', 
         '$W/m^2$',
         'Rot. per Minute (RPM)',
         'Celsius',
         'V',
         'A']

for i in range(len(titles)):
    print(feats[i] + '\t' + titles[i] + '\t' + units[i])
print('\n')
#%% GPU INIT
#export TF_CUDNN_RESET_RND_GEN_STATE=1 
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
#%%
def print_line():
    print("\n\n#-------------------------------------------------------------------------------")

#%%
def compose_df (src_path):
    """
    Compose dataset from all '.csv' files in src_path

    Parameters
    ----------
    src_path : string
        path to folder containing csv files.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing the concatenated .csv files data.

    """
    flst = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.endswith('.csv')]
    df_from_each_file = (pd.read_csv(f) for f in flst)
    return pd.concat(df_from_each_file, ignore_index=True)
    # df_scaled, df_mean, df_std = scaleDataframe(df)
    ## Export data
    # df.to_pickle(dst_filename)
    # df_scaled.to_pickle(dst_filename + '_scaled')
    # df_mean.to_pickle(dst_filename + '_mean')
    # df_std.to_pickle(dst_filename + '_std')

# %%
def scale_dataframe(df):
    """
    Standardize dataframe so that each Series has zero mean and std. dev = 1.0.
    Return also the mean and standard deviation of the original dataset.

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    df_scaled : pandas.DataFrame
        A pandas dataframe containing the scaled data.
    """
    scaler = preprocessing.StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df.values), columns=list(df)), df.mean(), df.std()

#%%
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

#%% Plot properties
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)

boxprops = dict(linestyle='-',
                linewidth=3,
                color='red')
flierprops = dict(marker='o',
                  markerfacecolor='green',
                  markersize=12,
                  linestyle='none')
medianprops = dict(linestyle='-',
                   linewidth=4,
                   color='black')
meanpointprops = dict(marker='D',
                      markeredgecolor='green',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='-',
                     linewidth=4, color='blue')
whiskerprops = dict(linestyle='-',
                    linewidth=3,
                    color='red')

def plot_df_columns(df):
    for column in df.columns:
        if column == "TIMESTAMP":
            continue
        fig = plt.figure(figsize=[14,6])
        plt.subplot(1, 2, 1)
        df[column].plot()
        plt.subplot(1, 2, 2)
        df[[column]].boxplot(meanprops=meanpointprops,
                            medianprops=medianprops,
                            showmeans=True,
                            meanline=False,
                            whiskerprops=whiskerprops,
                            boxprops=boxprops)
        plt.xticks([])
        fig.suptitle(column, fontsize=22)
        plt.title(column)
        plt.show()

def plot_df_histograms(df, save_folder=None):
    for column in df.columns:
        if column == "TIMESTAMP":
            continue
        fig = plt.figure(figsize=[14,6])
        plt.subplot(1, 2, 1)
        df[column].plot()
        plt.subplot(1, 2, 2)
        df[column].plot.hist(bins=100)
        plt.xticks([])
        fig.suptitle(column, fontsize=22)
        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, column + ".png"), dpi=300)
        plt.show()
        

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

#%%
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
    	
def generate_train_test_valid(folder, input_labels, output_labels, n_steps_in, n_steps_out, n_folds, day_only):
    
    """
    Divides the complete dataset into training, validation and test sets.
    Saves the training and test datasets in completion and then re divides the 
    training set in training and validation set using K fold validation and saves which fold.
    
    
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
        Number of future data which will represent a single output seuqence.
        How many minutes we will want to predict.
    n_folds : Integer
        Number of folds for K fold validation 
    input_labels : List
        List of String containnning the labels of the input of the model 
    output_labels : List
        List of String containnning the labels of the output of the model
    day_only : Bool
        True to remove the night values, False to leave them.
    Returns
    -------
    Nothing
     
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
    
    #Transforming the TIMESTAMP time values into floats. If it is in the input labels.
    if 'TIMESTAMP' in input_labels:
        aux_column = []
        for i in range(len(df_complete['TIMESTAMP'])):
            aux_column.append(df_complete['TIMESTAMP'][i][11:13]+df_complete['TIMESTAMP'][i][14:16])
        df_aux = pd.DataFrame(aux_column).astype(float)
        df_complete = df_complete.assign(TIMESTAMP = df_aux)
        
    
    #Removing not a number from the dataset
    df_label = df_complete[input_labels]
    df_label = df_label.dropna()
    df_label = df_label.reset_index(drop=True)
    #Negative numbers turn into 0 for each label
    for i in input_labels:
        indexAux = df_label[(df_label[i] < 0)].index
        df_label[i][indexAux] = 0.0
        normalizator = preprocessing.MinMaxScaler(feature_range=(0,1))
        normalizator.fit(df_label[i].values.reshape(-1, 1))
        df_label[i] = normalizator.transform(df_label[i].values.reshape(-1, 1))
        joblib.dump(normalizator, r'./../saves/norm/norm'+str(i)+'.save')

    
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
    pd.DataFrame(inputData).to_csv((r'./../folds/testInputData.csv'), index = False)
    pd.DataFrame(outputData).to_csv((r'./../folds/testOutputData.csv'), index = False)

    #Splitting the training data into input and output.
    inputData, outputData = split_sequence(trainingData, n_steps_in, n_steps_out, input_labels, output_labels)
    if n_steps_in > 1:    
        inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
    if n_steps_out > 1:
        outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])
    pd.DataFrame(inputData).to_csv((r'./../folds/trainingInputData.csv'), index = False)
    pd.DataFrame(outputData).to_csv((r'./../folds/trainingOutputData.csv'), index = False)
    
    #Splitting the training data in taining and validation folds
    kf = KFold(n_splits=n_folds)
    fold = 1
    for training_index, test_index in kf.split(inputData, outputData):
       training_input_fold = inputData[training_index[0]:training_index[len(training_index)-1]]
       pd.DataFrame(training_input_fold).to_csv((r'./../folds/trainingInputFold['+str(fold)+'].csv'), index = False)
       
       training_output_fold = outputData[training_index[0]:training_index[len(training_index)-1]]
       pd.DataFrame(training_output_fold).to_csv((r'./../folds/trainingOutputFold['+str(fold)+'].csv'), index = False)
      
       test_input_fold = inputData[test_index[0]:test_index[len(test_index)-1]]
       pd.DataFrame(test_input_fold).to_csv((r'./../folds/validationInputFold['+str(fold)+'].csv'), index = False)
       
       test_output_fold = outputData[test_index[0]:test_index[len(test_index)-1]]
       pd.DataFrame(test_output_fold).to_csv((r'./../folds/validationOutputFold['+str(fold)+'].csv'), index = False)
       
       fold = fold + 1
#%%
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
    return backend.mean(backend.abs(tf.math.subtract(y_true, y_pred)), axis =0)
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
    return backend.sqrt(backend.mean(backend.square(tf.math.subtract(y_true, y_pred)), axis =0))
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
    return tf.keras.backend.std(tf.math.subtract(y_true, y_pred), axis =0)

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
    #Computing the mean absolute error (MAE) between real Avarage PV Power and the predictions using tensorflow
    #mae = tf.keras.losses.MAE(y, y_hat)
    mae = mae_multi(y, y_hat)
    #Computing the root mean square (RMS) error between real Avarage PV Power and the predictions using tensorflow
    #rmse = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y, y_hat)))
    #rmse = tf.sqrt(tf.keras.metrics.MSE(y, y_hat))
    rmse = root_mean_square_error(y, y_hat)
    #Computing the standart deviation of the error of prediction using tensorflow
    stddev = standard_deviation_error(y, y_hat)
    #stddev_p2 = statistics.stdev(tf.math.subtract(y, y_hat).numpy())
    
    return mae.numpy(), rmse.numpy(), stddev.numpy()
#%%
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
    
def dispersion_plot(y_true,y_pred, save_path):
    """
    Plot and saves a dispersion plot between the real value and the predicted value.
    
    Parameters
    ----------
    y_true : Series
        Containing the real value.
    y_pred : Integer
        Containing the predict value.  
    save_path : String
        String containing the path to save the model.
    Returns
    -------
   
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    plot_size = y_true.shape[1]
    for i in range(plot_size):
        minute = i + 1 
        fig = plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(1, 1, 1)
        ax1.scatter(y_true[:,i], y_pred[:,i], s=1, c='b')
        ax1.plot(y_true[:,i], y_true[:,i], color = 'r')
        ax1.set_ylabel("Real", fontsize = 13)
        ax1.set_xlabel("Previsto", fontsize = 13)
        plt.title("Gráfico de Dispersão - Minuto "+str(minute), fontsize = 13)
        plt.savefig(save_path, dpi=300)
        
    return
def predictions_plot(y_true,y_pred, save_path):
    """
    Plot and saves plot the real value and the predicted value x Timesteps.
    
    Parameters
    ----------
    y_true : Series
        Containing the real value.
    y_pred : Integer
        Containing the predict value. 

    save_path : String
        String containing the path to save the model.
    Returns
    -------
   
    """    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    plot_size = y_true.shape[1]
    for i in range(plot_size):
        minute = i + 1
        fig = plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(1, 1, 1)
        ax1.plot(y_true[:,i],linestyle='-',color= 'red',label = 'Real', linewidth=1)
        ax1.plot(y_pred[:,i],linestyle='--', color= 'royalblue', label = 'Prevista', linewidth=1,dashes=(1, 2))
        ax1.set_ylabel("Unit", fontsize = 18)
        ax1.set_xlabel("Timestep", fontsize = 18)
        plt.title("Gráfico Real x Predito - Minuto "+str(minute), fontsize = 18)
        plt.legend()
        plt.grid(b=True)
        plt.savefig(save_path, dpi=300)
    return
#%% MLP


def mlp_kfold(n_steps_in,n_steps_out, input_labels, n_folds, layers_list):
    """
    Creates a MLP model to predict the labels and uses K Fold validation to evaluate the model.
    Saves which history of training in: ../saves/history_MLP['+str(fold)+']'+layer1+layer2+'_'+labels+'.
    Saves which mlp model in: ../saves/regressor_MLP['+str(fold)+']'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------

    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict
    input_labels : List
        List of String containnning the labels of the input of the model.
    n_folds : Integer
        Number of folds for K fold validation
    layers_list : List
        List containing the number of neurons on each layer. 
    Returns
    -------
    Nothing
    """
    
    for fold in range(1,n_folds+1):
        #Reading the training input data
        base_input_training = pd.read_csv(r'./../folds/trainingInputFold['+str(fold)+'].csv')
        #Getting the column values
        base_input_training = base_input_training.values
        #Reading the training output data
        base_output_training = pd.read_csv(r'./../folds/trainingOutputFold['+str(fold)+'].csv')
        base_output_training = base_output_training.values
        
        
        #Reading the test database for trainning validation
        base_input_test = pd.read_csv(r'./../folds/validationInputFold'+'['+str(fold)+'].csv') 
        base_input_test = base_input_test.values  
        base_output_test = pd.read_csv(r'./../folds/validationOutputFold'+'['+str(fold)+'].csv')
        base_output_test = base_output_test.values
        
        save_path = ""
        inputDim = n_steps_in * len(input_labels) 
        
        
        #define model
        model = Sequential()
        for i in range(len(layers_list)):
            if (layers_list[i] > 0) and i == 0:     
               model.add(Dense(units = layers_list[i], activation='relu', input_dim=inputDim))
            elif (layers_list[i] > 0):
                model.add(Dense(units = layers_list[i], activation='relu'))
            save_path = save_path + "["+str(layers_list[i])+"]"
        save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
        for i in range(len(input_labels)):
             save_path = save_path +"["+str(input_labels[i]) + "]"
        # Output layer
        model.add(Dense(units = n_steps_out, activation = 'linear'))
        # Compilling the network according to the loss_metric
        opt = optimizers.Adam(lr=0.001)
        model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])
        # função early stop vai parar de treinar a rede se algum parâmetro monitorado parou de melhorar
        es = EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 10, verbose = 1)
        # Reduce the learnning rate when the metric stop improving.
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, verbose = 1)
        mcp =  ModelCheckpoint(filepath=r'./../saves/mse/pesos_MLP['+str(fold)+']'+save_path+'.h5', monitor = 'val_loss', save_best_only= True)
            
        #training and storing the history
        history = model.fit(x = base_input_training,
                                y= base_output_training, 
                                validation_data = (base_input_test, base_output_test),
                                epochs = 128,
                                batch_size = 512,
                                callbacks = [es,rlr, mcp])
        
        model_json = model.to_json()
        hist = {'loss': str(history.history['loss']),
                'val_loss': str(history.history['val_loss']),
                'mae': str(history.history['mae_multi']),
                'val_mae': str(history.history['val_mae_multi']),
                'rmse': str(history.history['root_mean_square_error']),
                'val_rmse': str(history.history['val_root_mean_square_error']),
                'stddev': str(history.history['standard_deviation_error']),
                'val_stddev': str(history.history['val_standard_deviation_error'])
                }
        j_hist = json.dumps(hist)
        with open(r'./../saves/mse/history_MLP['+str(fold)+']'+save_path, 'w') as json_file:
            json_file.write(j_hist)
        with open(r'./../saves/mse/regressor_MLP['+str(fold)+']'+save_path+'.json', 'w') as json_file:
            json_file.write(model_json)

  
def evaluate_mlp_history(n_steps_in, n_steps_out, input_labels, n_folds, layers_list):
    """
    Gets the trainning and Validation data from the history and
    plots the history of training of the MLP. 
    
    Parameters
    ----------     
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict
    input_labels : List
        List of String containnning the labels of the input of the model.
    n_folds : Integer
        Number of folds for K fold validation 
    layers_list : List
        List containing the number of neurons on each layer.
    Returns
    -------
    evaluation : dict
        Dictonary containing: loss, val_loss, mae, val_mae,
        rmse, val_rmse, stddev and val_stddev.   
    """
    loss     = []
    val_loss = []
    mae = []
    val_mae = []
    rmse = []
    val_rmse = []
    stddev = []
    val_stddev = []
    
    save_path = ""
    for i in range(len(layers_list)):
        save_path = save_path + "["+str(layers_list[i])+"]"
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
             save_path = save_path +"["+str(input_labels[i]) + "]"
    for j in range(1,9):
        fold= j
        aux_loss = []
        aux_val_loss = []
        aux_mae = []
        aux_val_mae = []
        aux_rmse = []
        aux_val_rmse = []
        aux_stddev = []
        aux_val_stddev = []
        
        with open(r'./../saves/mse/history_MLP['+str(fold)+']'+save_path) as f:
            js = json.load(f)
            aux = js.get("loss")
            aux = aux.split(", ")
            aux_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_loss.append(float(aux[i]))
            aux_loss.append(float(aux[i+1].replace("]", "")))
                
            aux = js.get("val_loss")
            aux = aux.split(", ")
            aux_val_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_loss.append(float(aux[i]))
            aux_val_loss.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("mae")
            aux = aux.split(", ")
            aux_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_mae.append(float(aux[i]))
            aux_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_mae")
            aux = aux.split(", ")
            aux_val_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_mae.append(float(aux[i]))
            aux_val_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("rmse")
            aux = aux.split(", ")
            aux_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_rmse.append(float(aux[i]))
            aux_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_rmse")
            aux = aux.split(", ")
            aux_val_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_rmse.append(float(aux[i]))
            aux_val_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("stddev")
            aux = aux.split(", ")
            aux_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_stddev.append(float(aux[i]))
            aux_stddev.append(float(aux[i+1].replace("]", "")))
            
            
            aux = js.get("val_stddev")
            aux = aux.split(", ")
            aux_val_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_stddev.append(float(aux[i]))
            aux_val_stddev.append(float(aux[i+1].replace("]", "")))

        loss.append(aux_loss)
        val_loss.append(aux_val_loss)
        mae.append(aux_mae)
        val_mae.append(aux_val_mae)
        rmse.append(aux_rmse)
        val_rmse.append(aux_val_rmse)
        stddev.append(aux_stddev)
        val_stddev.append(aux_val_stddev)
    
    
    mean_loss = []
    mean_val_loss = []
    last_mean_loss = []
    last_mean_val_loss = []
    last_mean_mae = []
    last_mean_val_mae = []
    last_mean_rmse = []
    last_mean_val_rmse = []
    last_mean_stddev = []
    last_mean_val_stddev = []
    
    
    for i in range(n_folds):
        last_mean_loss.append(loss[i][len(loss[i])-1])
        last_mean_val_loss.append(val_loss[i][len(loss[i])-1])
        last_mean_mae.append(mae[i][len(mae[i])-1])
        last_mean_val_mae.append(val_mae[i][len(val_mae[i])-1])
        last_mean_rmse.append(rmse[i][len(rmse[i])-1])
        last_mean_val_rmse.append(val_rmse[i][len(val_rmse[i])-1])
        last_mean_stddev.append(stddev[i][len(stddev[i])-1])
        last_mean_val_stddev.append(val_stddev[i][len(val_stddev[i])-1])
    
    last_mean_loss = tf.reduce_mean(last_mean_loss).numpy()
    last_mean_val_loss = tf.reduce_mean(last_mean_val_loss).numpy()
    last_mean_mae = tf.reduce_mean(last_mean_mae).numpy()
    last_mean_val_mae = tf.reduce_mean(last_mean_val_mae).numpy()
    last_mean_rmse = (tf.reduce_mean(last_mean_rmse)).numpy()
    last_mean_val_rmse = (tf.reduce_mean(last_mean_val_rmse)).numpy()
    last_mean_stddev = (tf.reduce_mean(last_mean_stddev)).numpy()
    last_mean_val_stddev = (tf.reduce_mean(last_mean_val_stddev)).numpy()
    
    evaluation = {'loss': last_mean_loss,
                'val_loss': last_mean_val_loss,
                'mae': last_mean_mae,
                'val_mae': last_mean_val_mae,
                'rmse': last_mean_rmse,
                'val_rmse': last_mean_val_rmse,
                'stddev': last_mean_stddev,
                'val_stddev': last_mean_val_stddev}
    
    for i in range(len(loss[0])):
        aux = []
        aux2 = []
        for j in range(n_folds):
            if  i < len(loss[j]):
                aux.append(loss[j][i])
                aux2.append(val_loss[j][i]) 
        mean_loss.append((tf.reduce_mean(aux)).numpy())
        mean_val_loss.append((tf.reduce_mean(aux2)).numpy())
    
    print("Evaluation of trainning and validation:")
    print("loss: ", last_mean_loss, "val_loss: ", last_mean_val_loss,"\n", "mae: ", last_mean_mae, 
          "val_mae: ", last_mean_val_mae, "\n" ,"rmse: ", last_mean_rmse, "val_rmse: ", last_mean_val_rmse,"\n" 
          "stddev: ", last_mean_stddev,"val_stddev: ", last_mean_val_stddev)
    
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(mean_loss, label = "Treinamento")
    ax1.plot(mean_val_loss, label = "Validação")
    ax1.set_ylabel("Média do Erro Quadrático Médio", fontsize = 13)
    ax1.set_xlabel("Épocas", fontsize = 13)
    plt.title("MLP"+save_path, fontsize=8)
    plt.legend()    
    plt.savefig(r'./../res/mlp/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/MLP'+save_path+'.png', dpi=300)
    
    return evaluation

def mlp_univariate_full(n_steps_in, n_steps_out, input_labels, layers_list):
    """
    Creates a MLP model to predict the labels and uses all the data for training.
    Saves which history of training in: ../saves/history_MLP'+layer1+layer2+'_'+labels+'.
    Saves which mlp model in: ../saves/regressor_MLP'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    input_labels : List
        List of String containnning the labels of the input of the model.
    layers_list : List
        List containing the number of neurons on each layer.

    Returns
    -------
    Nothing
    """
   
    #Training the model with all the training data
    #Reading the training input data
    input_training = pd.read_csv(r'./../folds/trainingInputData.csv')
    #Getting the column values
    input_training = input_training.values
    #Reading the training output data
    output_training = pd.read_csv(r'./../folds/trainingOutputData.csv')
    output_training = output_training.values
    
    save_path = ""
    inputDim = n_steps_in * len(input_labels) 
    #define model
    model = Sequential()
    for i in range(len(layers_list)):
        if (layers_list[i] > 0) and i == 0:     
           model.add(Dense(units = layers_list[i], activation='relu', input_dim=inputDim))
        elif (layers_list[i] > 0):
            model.add(Dense(units = layers_list[i], activation='relu'))
        save_path = save_path + "["+str(layers_list[i])+"]"
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
             save_path = save_path +"["+str(input_labels[i]) + "]"
    # Output layer
    model.add(Dense(units = n_steps_out, activation = 'linear'))
    # Compilling the network according to the loss_metric
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])  
    es = EarlyStopping(monitor ='loss', min_delta = 1e-9, patience = 10, verbose = 1)
    # Reduce the learnning rate when the metric stop improving.
    rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 5, verbose = 1)
    mcp =  ModelCheckpoint(filepath=r'./../saves/mse/pesos_MLP'+save_path+'.h5', monitor = 'loss', save_best_only= True)
    #training and storing the history
    history = model.fit(x = input_training,
                            y= output_training, 
                            epochs = 128,
                            batch_size = 512,
                            callbacks = [es,rlr,mcp])
    model_json = model.to_json()
   
         
    hist = {'loss': str(history.history['loss']),
            'mae': str(history.history['mae_multi']),
            'rmse': str(history.history['root_mean_square_error']),
            'stddev': str(history.history['standard_deviation_error'])
            }
    j_hist = json.dumps(hist)
    with open(r'./../saves/mse/history_MLP'+save_path, 'w') as json_file:
        json_file.write(j_hist)
    with open(r'./../saves/mse/regressor_MLP'+save_path+'.json', 'w') as json_file:
       json_file.write(model_json)




def mlp_univariate_test(n_steps_in, n_steps_out, input_labels, layers_list):
    """
    Testing the fully trained model with the test data.
    Displays the results in form of: Mean Absolute Error (MAE), 
    Root Mean Square Error (RMS) and Standart Deviation.
    
    Parameters
    ----------
    labels : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    layers_list : List
        List containing the number of neurons on each layer.
    Returns
    -------
    mae : list
        Mean Absolute Error for each output.
    rms : list
        Root Mean Square for each output.
    stddev : list
        Standart Deviation for each output.
    """
    #Testing the model fully trained
    #Reading the test input data
    input_test = pd.read_csv(r'./../folds/testInputData.csv')
    #Getting the column values
    input_test = input_test.values
    #Reading the training output data
    output_test = pd.read_csv(r'./../folds/testOutputData.csv')
    output_test = output_test.values
    
    save_path = ""
    for i in range(len(layers_list)):
        save_path = save_path + "["+str(layers_list[i])+"]"
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
        save_path = save_path + "["+str(input_labels[i]) + "]" 
    # Openning the file which contains the network model
    mlp_file = open(r'./../saves/mse/regressor_MLP'+save_path+'.json', 'r')
    mlp_structure = mlp_file.read()
    # Fechando o arquivo
    mlp_file.close()
    # Getting the network structure
    model = model_from_json(mlp_structure)
    # Reading the weights the putting them  in the network model
    model.load_weights(r'./../saves/mse/pesos_MLP'+save_path+'.h5')

    predictions = model.predict(input_test)
    
    normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')

    y = normalizator.inverse_transform(output_test)
    y_hat = normalizator.inverse_transform(predictions)    

    mae, rmse, stddev = mae_rmse_stddev_evaluation(y, y_hat)
    plot_save_path = r'./../res/mlp/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/GraficoDispersaoMLP'+save_path+'.png'
    dispersion_plot(y, y_hat, plot_save_path)
    plot_save_path = r'./../res/mlp/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/Grafico_RealXPreditoMLP'+save_path+'.png'
    predictions_plot(y, y_hat, plot_save_path)
    
    print("Tests evaluation:")
    for i in range(len(mae)):
        print(str(i+1)+" Output minute:")
        print("Mean Absolute Error (MAE): {:.3f}".format(mae[i]))
        print("Root Mean Square Error (RMS): {:.3f}".format(rmse[i]))
        print("Standart Deviation: {:.3f}".format(stddev[i]))

    
    return mae, rmse, stddev
#%% RNN


def rnn_kfold(n_steps_in,n_steps_out, input_labels, n_folds, layers_list):
    """
    Creates a RNN model to predict the labels and uses K Fold validation to evaluate the model.
    Saves which history of training in: ../saves/history_RNN['+str(fold)+']'+layer1+layer2+'_'+labels+'.
    Saves which RNN model in: ../saves/regressor_RNN['+str(fold)+']'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
  
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict
    input_labels : List
        List of String containnning the labels of the input of the model.
    n_folds : Integer
        Number of folds for K fold validation
    layers_list : List
        List containing the number of neurons on each layer. 
    Returns
    -------
    Nothing
    """
    
    for fold in range(1,n_folds+1):
        #Reading the training input data
        X_train  = pd.read_csv(r'./../folds/trainingInputFold['+str(fold)+'].csv')
        #Getting the column values
        X_train  = X_train.values
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
        #Reading the training output data
        Y_train= pd.read_csv(r'./../folds/trainingOutputFold['+str(fold)+'].csv')
        Y_train = Y_train.values
        

        
        #Reading the test database for trainning validation
        X_test  = pd.read_csv(r'./../folds/validationInputFold'+'['+str(fold)+'].csv') 
        X_test  = X_test.values 
        X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)
        Y_test = pd.read_csv(r'./../folds/validationOutputFold'+'['+str(fold)+'].csv')
        Y_test = Y_test.values
        
        save_path = ""
        save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
        for i in range(len(input_labels)):
             save_path = save_path +"["+str(input_labels[i]) + "]"
        #define model
        model = Sequential()
        model.add(SimpleRNN(units = 32, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
        model.add(SimpleRNN(units = 32, activation="tanh", return_sequences = True))
        model.add(SimpleRNN(units = 256, activation="tanh", return_sequences = True))
        model.add(SimpleRNN(units = 32, activation="tanh", return_sequences = True))
        model.add(SimpleRNN(units = 256, activation="tanh", return_sequences = True))
        model.add(SimpleRNN(units = 16, activation="tanh"))
        model.add(Dense(units = n_steps_out, activation = 'linear'))
        
        opt = optimizers.Adam(lr=0.0001)
        model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])
        es = EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 10, verbose = 1)
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, verbose = 1)
        mcp =  ModelCheckpoint(filepath=r'./../saves/mse/pesos_RNN['+str(fold)+']'+save_path+'.h5', monitor = 'val_loss', save_best_only= True)
            
        #training and storing the history
        history = model.fit(x = X_train,
                                y= Y_train, 
                                validation_data = (X_test, Y_test),
                                epochs = 128,
                                batch_size = 512,
                                callbacks = [es,rlr, mcp])
        
        model_json = model.to_json()
        hist = {'loss': str(history.history['loss']),
                'val_loss': str(history.history['val_loss']),
                'mae': str(history.history['mae_multi']),
                'val_mae': str(history.history['val_mae_multi']),
                'rmse': str(history.history['root_mean_square_error']),
                'val_rmse': str(history.history['val_root_mean_square_error']),
                'stddev': str(history.history['standard_deviation_error']),
                'val_stddev': str(history.history['val_standard_deviation_error'])
                }
        j_hist = json.dumps(hist)
        with open(r'./../saves/mse/history_RNN['+str(fold)+']'+save_path, 'w') as json_file:
            json_file.write(j_hist)
        with open(r'./../saves/mse/regressor_RNN['+str(fold)+']'+save_path+'.json', 'w') as json_file:
            json_file.write(model_json)

def evaluate_rnn_history(n_steps_in, n_steps_out, input_labels, n_folds, layers_list):
    """
    Creates a RNN model to predict the labels and uses all the data for training.
    Saves which history of training in: ../saves/history_RNN'+layer1+layer2+'_'+labels+'.
    Saves which RNN model in: ../saves/regressor_RNN'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    input_labels : List
        List of String containnning the labels of the input of the model.
    layers_list : List
        List containing the number of neurons on each layer.

    Returns
    -------
    Nothing
    """
    loss     = []
    val_loss = []
    mae = []
    val_mae = []
    rmse = []
    val_rmse = []
    stddev = []
    val_stddev = []
    
    save_path = ""
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
         save_path = save_path +"["+str(input_labels[i]) + "]"
    
    for j in range(1,9):
        fold= j
        aux_loss = []
        aux_val_loss = []
        aux_mae = []
        aux_val_mae = []
        aux_rmse = []
        aux_val_rmse = []
        aux_stddev = []
        aux_val_stddev = []
        
        with open(r'./../saves/mse/history_RNN['+str(fold)+']'+save_path) as f:
            js = json.load(f)
            aux = js.get("loss")
            aux = aux.split(", ")
            aux_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_loss.append(float(aux[i]))
            aux_loss.append(float(aux[i+1].replace("]", "")))
                
            aux = js.get("val_loss")
            aux = aux.split(", ")
            aux_val_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_loss.append(float(aux[i]))
            aux_val_loss.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("mae")
            aux = aux.split(", ")
            aux_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_mae.append(float(aux[i]))
            aux_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_mae")
            aux = aux.split(", ")
            aux_val_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_mae.append(float(aux[i]))
            aux_val_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("rmse")
            aux = aux.split(", ")
            aux_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_rmse.append(float(aux[i]))
            aux_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_rmse")
            aux = aux.split(", ")
            aux_val_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_rmse.append(float(aux[i]))
            aux_val_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("stddev")
            aux = aux.split(", ")
            aux_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_stddev.append(float(aux[i]))
            aux_stddev.append(float(aux[i+1].replace("]", "")))
            
            
            aux = js.get("val_stddev")
            aux = aux.split(", ")
            aux_val_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_stddev.append(float(aux[i]))
            aux_val_stddev.append(float(aux[i+1].replace("]", "")))
    
        loss.append(aux_loss)
        val_loss.append(aux_val_loss)
        mae.append(aux_mae)
        val_mae.append(aux_val_mae)
        rmse.append(aux_rmse)
        val_rmse.append(aux_val_rmse)
        stddev.append(aux_stddev)
        val_stddev.append(aux_val_stddev)
    
    
    mean_loss = []
    mean_val_loss = []
    last_mean_loss = []
    last_mean_val_loss = []
    last_mean_mae = []
    last_mean_val_mae = []
    last_mean_rmse = []
    last_mean_val_rmse = []
    last_mean_stddev = []
    last_mean_val_stddev = []
    
    
    for i in range(n_folds):
        last_mean_loss.append(loss[i][len(loss[i])-1])
        last_mean_val_loss.append(val_loss[i][len(loss[i])-1])
        last_mean_mae.append(mae[i][len(mae[i])-1])
        last_mean_val_mae.append(val_mae[i][len(val_mae[i])-1])
        last_mean_rmse.append(rmse[i][len(rmse[i])-1])
        last_mean_val_rmse.append(val_rmse[i][len(val_rmse[i])-1])
        last_mean_stddev.append(stddev[i][len(stddev[i])-1])
        last_mean_val_stddev.append(val_stddev[i][len(val_stddev[i])-1])
    
    last_mean_loss = tf.reduce_mean(last_mean_loss).numpy()
    last_mean_val_loss = tf.reduce_mean(last_mean_val_loss).numpy()
    last_mean_mae = tf.reduce_mean(last_mean_mae).numpy()
    last_mean_val_mae = tf.reduce_mean(last_mean_val_mae).numpy()
    last_mean_rmse = (tf.reduce_mean(last_mean_rmse)).numpy()
    last_mean_val_rmse = (tf.reduce_mean(last_mean_val_rmse)).numpy()
    last_mean_stddev = (tf.reduce_mean(last_mean_stddev)).numpy()
    last_mean_val_stddev = (tf.reduce_mean(last_mean_val_stddev)).numpy()
    
    evaluation = {'loss': last_mean_loss,
                'val_loss': last_mean_val_loss,
                'mae': last_mean_mae,
                'val_mae': last_mean_val_mae,
                'rmse': last_mean_rmse,
                'val_rmse': last_mean_val_rmse,
                'stddev': last_mean_stddev,
                'val_stddev': last_mean_val_stddev}
    
    for i in range(len(loss[0])):
        aux = []
        aux2 = []
        for j in range(n_folds):
            if  i < len(loss[j]):
                aux.append(loss[j][i])
                aux2.append(val_loss[j][i]) 
        mean_loss.append((tf.reduce_mean(aux)).numpy())
        mean_val_loss.append((tf.reduce_mean(aux2)).numpy())
    
    print("Evaluation of trainning and validation:")
    print("loss: ", last_mean_loss, "val_loss: ", last_mean_val_loss,"\n", "mae: ", last_mean_mae, 
          "val_mae: ", last_mean_val_mae, "\n" ,"rmse: ", last_mean_rmse, "val_rmse: ", last_mean_val_rmse,"\n" 
          "stddev: ", last_mean_stddev,"val_stddev: ", last_mean_val_stddev)
    
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(mean_loss, label = "Treinamento")
    ax1.plot(mean_val_loss, label = "Validação")
    ax1.set_ylabel("Média do Erro Quadrático Médio", fontsize = 13)
    ax1.set_xlabel("Épocas", fontsize = 13)
    plt.title("RNN"+save_path, fontsize=8)
    plt.legend()    
    plt.savefig(r'./../res/rnn/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/RNN'+save_path+'.png', dpi=300)
    return evaluation
def rnn_full(n_steps_in, n_steps_out, input_labels, layers_list):
    """
    Creates a RNN model to predict the labels and uses all the data for training.
    Saves which history of training in: ../saves/mse/history_RNN'+layer1+layer2+'_'+labels+'.
    Saves which RNN model in: ../saves/mse/regressor_RNN'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    input_labels : List
        List of String containnning the labels of the input of the model.
    layers_list : List
        List containing the number of neurons on each layer.

    Returns
    -------
    Nothing
    """
    X_train = pd.read_csv(r'./../folds/trainingInputData.csv')
    X_train = X_train.values
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
    
    Y_train = pd.read_csv(r'./../folds/trainingOutputData.csv')
    Y_train = Y_train.values
    
    
 
    save_path = ""
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
         save_path = save_path +"["+str(input_labels[i]) + "]"
    model = Sequential()
    model.add(SimpleRNN(units = 32, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
    model.add(SimpleRNN(units = 32, activation="tanh", return_sequences = True))
    model.add(SimpleRNN(units = 256, activation="tanh", return_sequences = True))
    model.add(SimpleRNN(units = 32, activation="tanh", return_sequences = True))
    model.add(SimpleRNN(units = 256, activation="tanh", return_sequences = True))
    model.add(SimpleRNN(units = 16, activation="tanh"))
    model.add(Dense(units = n_steps_out, activation = 'linear'))
    
    
    # Compilling the network according to the loss_metric
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])  
    es = EarlyStopping(monitor ='loss', min_delta = 1e-9, patience = 10, verbose = 1)
    # reduz a taxa de aprendizagem quando uma metrica parou de melhorar
    rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 5, verbose = 1)
    mcp =  ModelCheckpoint(filepath=r'./../saves/mse/pesos_RNN'+save_path+'.h5', monitor = 'loss', save_best_only= True)
    #training and storing the history
    history = model.fit(x = X_train,
                            y= Y_train, 
                            epochs = 128,
                            batch_size = 512,
                            callbacks = [es,rlr,mcp])
    model_json = model.to_json()
       
         
    hist = {'loss': str(history.history['loss']),
            'mae': str(history.history['mae_multi']),
            'rmse': str(history.history['root_mean_square_error']),
            'stddev': str(history.history['standard_deviation_error'])
            }
    j_hist = json.dumps(hist)
    with open(r'./../saves/mse/history_RNN'+save_path, 'w') as json_file:
        json_file.write(j_hist)
    with open(r'./../saves/mse/regressor_RNN'+save_path+'.json', 'w') as json_file:
       json_file.write(model_json)

def rnn_test(n_steps_in, n_steps_out, input_labels, layers_list):
    """
    Testing the fully trained model with the test data.
    Displays the results in form of: Mean Absolute Error (MAE), 
    Root Mean Square Error (RMS) and Standart Deviation.
    
    Parameters
    ----------
    labels : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    layers_list : List
        List containing the number of neurons on each layer.
    Returns
    -------
    mae : list
        Mean Absolute Error for each output.
    rms : list
        Root Mean Square for each output.
    stddev : list
        Standart Deviation for each output.
    """
    input_test = pd.read_csv(r'./../folds/testInputData.csv')
    
    input_test = input_test.values
    input_test = input_test.reshape(input_test.shape[0], input_test.shape[1],1)
    
    output_test = pd.read_csv(r'./../folds/testOutputData.csv')
    output_test = output_test.values
    
    save_path = ""
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
         save_path = save_path +"["+str(input_labels[i]) + "]"

    rnn_file = open(r'./../saves/mse/regressor_RNN'+save_path+'.json', 'r')
    rnn_structure = rnn_file.read()
    # Fechando o arquivo
    rnn_file.close()
    # Getting the network structure
    model = model_from_json(rnn_structure)
    # Reading the weights the putting them  in the network model
    model.load_weights(r'./../saves/mse/pesos_RNN'+save_path+'.h5')
    
    predictions = model.predict(input_test)
    
    normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')

    
    y = normalizator.inverse_transform(output_test)
    y_hat = normalizator.inverse_transform(predictions)
    
    mae, rmse, stddev = mae_rmse_stddev_evaluation(y, y_hat)
    plot_save_path = r'./../res/rnn/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/GraficoDispersaoRNN'+save_path+'.png'
    dispersion_plot(y, y_hat, plot_save_path)
    plot_save_path = r'./../res/rnn/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/Grafico_RealXPreditoRNN'+save_path+'.png'
    predictions_plot(y, y_hat, plot_save_path)
    
    print("Tests evaluation:")
    for i in range(len(mae)):
        print(str(i+1)+" Output minute:")
        print("Mean Absolute Error (MAE): {:.3f}".format(mae[i]))
        print("Root Mean Square Error (RMS): {:.3f}".format(rmse[i]))
        print("Standart Deviation: {:.3f}".format(stddev[i]))

    return mae, rmse, stddev



#%%
#LSTM

def lstm_kfold(n_steps_in,n_steps_out, input_labels, n_folds, layers_list):
    """
    Creates a lstm model to predict the labels and uses K Fold validation to evaluate the model.
    Saves which history of training in: ../saves/history_LSTM['+str(fold)+']'+layer1+layer2+'_'+labels+'.
    Saves which lstm
    model in: ../saves/regressor_LSTM['+str(fold)+']'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
  
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict
    input_labels : List
        List of String containnning the labels of the input of the model.
    n_folds : Integer
        Number of folds for K fold validation
    layers_list : List
        List containing the number of neurons on each layer. 
    Returns
    -------
    Nothing
    """
    
    for fold in range(1,n_folds+1):
        #Reading the training input data
        X_train  = pd.read_csv(r'./../folds/trainingInputFold['+str(fold)+'].csv')
        #Getting the column values
        X_train  = X_train.values
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
        #Reading the training output data
        Y_train= pd.read_csv(r'./../folds/trainingOutputFold['+str(fold)+'].csv')
        Y_train = Y_train.values
        

        
        #Reading the test database for trainning validation
        X_test  = pd.read_csv(r'./../folds/validationInputFold'+'['+str(fold)+'].csv') 
        X_test  = X_test.values 
        X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)
        Y_test = pd.read_csv(r'./../folds/validationOutputFold'+'['+str(fold)+'].csv')
        Y_test = Y_test.values
        
        save_path = ""
        save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
        for i in range(len(input_labels)):
             save_path = save_path +"["+str(input_labels[i]) + "]"
        #define model
        model = Sequential()
        model.add(LSTM(units = 32, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(units = 32, activation="tanh", return_sequences = True))
        model.add(LSTM(units = 256, activation="tanh", return_sequences = True))
        model.add(LSTM(units = 32, activation="tanh", return_sequences = True))
        model.add(LSTM(units = 256, activation="tanh", return_sequences = True))
        model.add(LSTM(units = 16, activation="tanh"))
        model.add(Dense(units = n_steps_out, activation = 'linear'))
        model.add(Dense(units = n_steps_out, activation = 'linear'))
        
        opt = optimizers.Adam(lr=0.0001)
        model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])
        es = EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 10, verbose = 1)
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, verbose = 1)
        mcp =  ModelCheckpoint(filepath=r'./../saves/mse/pesos_LSTM['+str(fold)+']'+save_path+'.h5', monitor = 'val_loss', save_best_only= True)
            
        #training and storing the history
        history = model.fit(x = X_train,
                                y= Y_train, 
                                validation_data = (X_test, Y_test),
                                epochs = 128,
                                batch_size = 512,
                                callbacks = [es,rlr, mcp])
        
        model_json = model.to_json()
        hist = {'loss': str(history.history['loss']),
                'val_loss': str(history.history['val_loss']),
                'mae': str(history.history['mae_multi']),
                'val_mae': str(history.history['val_mae_multi']),
                'rmse': str(history.history['root_mean_square_error']),
                'val_rmse': str(history.history['val_root_mean_square_error']),
                'stddev': str(history.history['standard_deviation_error']),
                'val_stddev': str(history.history['val_standard_deviation_error'])
                }
        j_hist = json.dumps(hist)
        with open(r'./../saves/mse/history_LSTM['+str(fold)+']'+save_path, 'w') as json_file:
            json_file.write(j_hist)
        with open(r'./../saves/mse/regressor_LSTM['+str(fold)+']'+save_path+'.json', 'w') as json_file:
            json_file.write(model_json)

def evaluate_lstm_history(n_steps_in, n_steps_out, input_labels, n_folds, layers_list):
    """
    Creates a LSTM model to predict the labels and uses all the data for training.
    Saves which history of training in: ../saves/history_LSTM'+layer1+layer2+'_'+labels+'.
    Saves which LSTM model in: ../saves/regressor_LSTM'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    input_labels : List
        List of String containnning the labels of the input of the model.
    layers_list : List
        List containing the number of neurons on each layer.

    Returns
    -------
    Nothing
    """
    loss     = []
    val_loss = []
    mae = []
    val_mae = []
    rmse = []
    val_rmse = []
    stddev = []
    val_stddev = []
    
    save_path = ""
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
         save_path = save_path +"["+str(input_labels[i]) + "]"
    
    for j in range(1,9):
        fold= j
        aux_loss = []
        aux_val_loss = []
        aux_mae = []
        aux_val_mae = []
        aux_rmse = []
        aux_val_rmse = []
        aux_stddev = []
        aux_val_stddev = []
        
        with open(r'./../saves/mse/history_LSTM['+str(fold)+']'+save_path) as f:
            js = json.load(f)
            aux = js.get("loss")
            aux = aux.split(", ")
            aux_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_loss.append(float(aux[i]))
            aux_loss.append(float(aux[i+1].replace("]", "")))
                
            aux = js.get("val_loss")
            aux = aux.split(", ")
            aux_val_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_loss.append(float(aux[i]))
            aux_val_loss.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("mae")
            aux = aux.split(", ")
            aux_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_mae.append(float(aux[i]))
            aux_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_mae")
            aux = aux.split(", ")
            aux_val_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_mae.append(float(aux[i]))
            aux_val_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("rmse")
            aux = aux.split(", ")
            aux_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_rmse.append(float(aux[i]))
            aux_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_rmse")
            aux = aux.split(", ")
            aux_val_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_rmse.append(float(aux[i]))
            aux_val_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("stddev")
            aux = aux.split(", ")
            aux_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_stddev.append(float(aux[i]))
            aux_stddev.append(float(aux[i+1].replace("]", "")))
            
            
            aux = js.get("val_stddev")
            aux = aux.split(", ")
            aux_val_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_stddev.append(float(aux[i]))
            aux_val_stddev.append(float(aux[i+1].replace("]", "")))
    
        loss.append(aux_loss)
        val_loss.append(aux_val_loss)
        mae.append(aux_mae)
        val_mae.append(aux_val_mae)
        rmse.append(aux_rmse)
        val_rmse.append(aux_val_rmse)
        stddev.append(aux_stddev)
        val_stddev.append(aux_val_stddev)
    
    
    mean_loss = []
    mean_val_loss = []
    last_mean_loss = []
    last_mean_val_loss = []
    last_mean_mae = []
    last_mean_val_mae = []
    last_mean_rmse = []
    last_mean_val_rmse = []
    last_mean_stddev = []
    last_mean_val_stddev = []
    
    
    for i in range(n_folds):
        last_mean_loss.append(loss[i][len(loss[i])-1])
        last_mean_val_loss.append(val_loss[i][len(loss[i])-1])
        last_mean_mae.append(mae[i][len(mae[i])-1])
        last_mean_val_mae.append(val_mae[i][len(val_mae[i])-1])
        last_mean_rmse.append(rmse[i][len(rmse[i])-1])
        last_mean_val_rmse.append(val_rmse[i][len(val_rmse[i])-1])
        last_mean_stddev.append(stddev[i][len(stddev[i])-1])
        last_mean_val_stddev.append(val_stddev[i][len(val_stddev[i])-1])
    
    last_mean_loss = tf.reduce_mean(last_mean_loss).numpy()
    last_mean_val_loss = tf.reduce_mean(last_mean_val_loss).numpy()
    last_mean_mae = tf.reduce_mean(last_mean_mae).numpy()
    last_mean_val_mae = tf.reduce_mean(last_mean_val_mae).numpy()
    last_mean_rmse = (tf.reduce_mean(last_mean_rmse)).numpy()
    last_mean_val_rmse = (tf.reduce_mean(last_mean_val_rmse)).numpy()
    last_mean_stddev = (tf.reduce_mean(last_mean_stddev)).numpy()
    last_mean_val_stddev = (tf.reduce_mean(last_mean_val_stddev)).numpy()
    
    evaluation = {'loss': last_mean_loss,
                'val_loss': last_mean_val_loss,
                'mae': last_mean_mae,
                'val_mae': last_mean_val_mae,
                'rmse': last_mean_rmse,
                'val_rmse': last_mean_val_rmse,
                'stddev': last_mean_stddev,
                'val_stddev': last_mean_val_stddev}
    
    for i in range(len(loss[0])):
        aux = []
        aux2 = []
        for j in range(n_folds):
            if  i < len(loss[j]):
                aux.append(loss[j][i])
                aux2.append(val_loss[j][i]) 
        mean_loss.append((tf.reduce_mean(aux)).numpy())
        mean_val_loss.append((tf.reduce_mean(aux2)).numpy())
    
    print("Evaluation of trainning and validation:")
    print("loss: ", last_mean_loss, "val_loss: ", last_mean_val_loss,"\n", "mae: ", last_mean_mae, 
          "val_mae: ", last_mean_val_mae, "\n" ,"rmse: ", last_mean_rmse, "val_rmse: ", last_mean_val_rmse,"\n" 
          "stddev: ", last_mean_stddev,"val_stddev: ", last_mean_val_stddev)
    
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(mean_loss, label = "Treinamento")
    ax1.plot(mean_val_loss, label = "Validação")
    ax1.set_ylabel("Média do Erro Quadrático Médio", fontsize = 13)
    ax1.set_xlabel("Épocas", fontsize = 13)
    plt.title("LSTM"+save_path, fontsize=8)
    plt.legend()    
    plt.savefig(r'./../res/lstm/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/LSTM'+save_path+'.png', dpi=300)
    return evaluation
def lstm_full(n_steps_in, n_steps_out, input_labels, layers_list):
    """
    Creates a LSTM model to predict the labels and uses all the data for training.
    Saves which history of training in: ../saves/mse/history_LSTM'+layer1+layer2+'_'+labels+'.
    Saves which LSTM model in: ../saves/mse/regressor_LSTM'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    input_labels : List
        List of String containnning the labels of the input of the model.
    layers_list : List
        List containing the number of neurons on each layer.

    Returns
    -------
    Nothing
    """
    X_train = pd.read_csv(r'./../folds/trainingInputData.csv')
    X_train = X_train.values
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
    
    Y_train = pd.read_csv(r'./../folds/trainingOutputData.csv')
    Y_train = Y_train.values
    
    
 
    save_path = ""
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
        save_path = save_path +"["+str(input_labels[i]) + "]"
    model = Sequential()
    #model.add(LSTM(units = 32, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(units = 32, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(units = 32, activation="tanh", return_sequences = True))
    model.add(LSTM(units = 256, activation="tanh", return_sequences = True))
    model.add(LSTM(units = 32, activation="tanh", return_sequences = True))
    model.add(LSTM(units = 256, activation="tanh", return_sequences = True))
    model.add(LSTM(units = 16, activation="tanh"))
    model.add(Dense(units = n_steps_out, activation = 'linear'))
    
    
    # Compilling the network according to the loss_metric
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])  
    es = EarlyStopping(monitor ='loss', min_delta = 1e-9, patience = 10, verbose = 1)
    # reduz a taxa de aprendizagem quando uma metrica parou de melhorar
    rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 5, verbose = 1)
    mcp =  ModelCheckpoint(filepath=r'./../saves/mse/pesos_LSTM'+save_path+'.h5', monitor = 'loss', save_best_only= True)
    #training and storing the history
    history = model.fit(x = X_train,
                            y= Y_train, 
                            epochs = 128,
                            batch_size = 512,
                            callbacks = [es,rlr,mcp])
    model_json = model.to_json()
       
         
    hist = {'loss': str(history.history['loss']),
            'mae': str(history.history['mae_multi']),
            'rmse': str(history.history['root_mean_square_error']),
            'stddev': str(history.history['standard_deviation_error'])
            }
    j_hist = json.dumps(hist)
    with open(r'./../saves/mse/history_LSTM'+save_path, 'w') as json_file:
        json_file.write(j_hist)
    with open(r'./../saves/mse/regressor_LSTM'+save_path+'.json', 'w') as json_file:
       json_file.write(model_json)

def lstm_test(n_steps_in, n_steps_out, input_labels, layers_list):
    """
    Testing the fully trained model with the test data.
    Displays the results in form of: Mean Absolute Error (MAE), 
    Root Mean Square Error (RMS) and Standart Deviation.
    
    Parameters
    ----------
    labels : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    layers_list : List
        List containing the number of neurons on each layer.
    Returns
    -------
    mae : list
        Mean Absolute Error for each output.
    rms : list
        Root Mean Square for each output.
    stddev : list
        Standart Deviation for each output.
    """
    input_test = pd.read_csv(r'./../folds/testInputData.csv')
    
    input_test = input_test.values
    input_test = input_test.reshape(input_test.shape[0], input_test.shape[1],1)
    
    output_test = pd.read_csv(r'./../folds/testOutputData.csv')
    output_test = output_test.values
    
    save_path = ""
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    for i in range(len(input_labels)):
         save_path = save_path +"["+str(input_labels[i]) + "]"

    lstm_file = open(r'./../saves/mse/regressor_LSTM'+save_path+'.json', 'r')
    lstm_structure = lstm_file.read()
    # Fechando o arquivo
    lstm_file.close()
    # Getting the network structure
    model = model_from_json(lstm_structure)
    # Reading the weights the putting them  in the network model
    model.load_weights(r'./../saves/mse/pesos_LSTM'+save_path+'.h5')
    
    predictions = model.predict(input_test)
    
    normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')

    
    y = normalizator.inverse_transform(output_test)
    y_hat = normalizator.inverse_transform(predictions)
    
    mae, rmse, stddev = mae_rmse_stddev_evaluation(y, y_hat)
    plot_save_path = r'./../res/lstm/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/GraficoDispersaoLSTM'+save_path+'.png'
    dispersion_plot(y, y_hat, plot_save_path)
    plot_save_path = r'./../res/lstm/mse/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/Grafico_RealXPreditoLSTM'+save_path+'.png'
    predictions_plot(y, y_hat, plot_save_path)
    
    print("Tests evaluation:")
    for i in range(len(mae)):
        print(str(i+1)+" Output minute:")
        print("Mean Absolute Error (MAE): {:.3f}".format(mae[i]))
        print("Root Mean Square Error (RMS): {:.3f}".format(rmse[i]))
        print("Standart Deviation: {:.3f}".format(stddev[i]))

    return mae, rmse, stddev

#%%

#MLP with correntropy
def correntropy(y_true, y_pred, kernel_size=1, a=1.0, b=0.0, clip=False):
    """
    Computes the parametric correntropy between two tensors y_true and y_pred.

    Inputs:     Both y_true and y_pred must have the same dimension.
                kernel_size is a scalar for the kernel size.
                a is a parameter for scaling y_true.
                b is a parameter to displace y_true.

    Outputs:    Parametric correntropy coefficient.

    Defaults:   a = 1.0
                b = 0.0

    Comments:   The code uses Incomplete Cholesky Decomposition.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)

    if clip:
        y_true = K.clip(y_true, K.epsilon(), 1.0)
        y_pred = K.clip(y_true, K.epsilon(), 1.0)

    y_true = a*y_true + b

    return tf.math.reduce_sum(tf.math.exp(-(y_true-y_pred)**2 / (2*kernel_size**2))) / tf.cast(tf.size(y_true), y_true.dtype)

def correntropy_loss(y_true, y_pred):
    """
    Computes the correntropy loss between two tensors y_true and y_pred.

    Inputs:     Both y_true and y_pred must have the same dimension.

    Outputs:    Correntropy loss.
    """
    return tf.convert_to_tensor(1.0 - correntropy(y_true, y_pred))     
 
def mlp_univariate_kfold_corr(label, n_steps_in, n_steps_out,n_folds, layers_list, kernel_size):
    """
    Using correntropy
    Creates a MLP model to predict the labels and uses K Fold validation to evaluate the model.
    Saves which history of training in: ../saves/history_MLP['+str(fold)+']'+layer1+layer2+'_'+labels+'.
    Saves which mlp model in: ../saves/regressor_MLP['+str(fold)+']'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    labels : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict
    n_folds : Integer
        Number of folds for K fold validation
    layers_list : List
        List containing the number of neurons on each layer. 
    kernel_size : float
        kernel_size is a scalar for the kernel size.
    Returns
    -------
    Nothing
    """
    
    for fold in range(1,n_folds+1):
        #Reading the training input data
        base_input_training = pd.read_csv(r'./../folds/trainingInputFold['+str(fold)+']_'+label+'.csv')
        #Getting the column values
        base_input_training = base_input_training.values
        #Reading the training output data
        base_output_training = pd.read_csv(r'./../folds/trainingOutputFold['+str(fold)+']_'+label+'.csv')
        base_output_training = base_output_training.values
        
        
        #Reading the test database for trainning validation
        base_input_test = pd.read_csv(r'./../folds/validationInputFold'+'['+str(fold)+']_'+label+'.csv') 
        base_input_test = base_input_test.values  
        base_output_test = pd.read_csv(r'./../folds/validationOutputFold'+'['+str(fold)+']_'+label+'.csv')
        base_output_test = base_output_test.values
        
        save_path = ""
        #define model
        model = Sequential()
        for i in range(len(layers_list)):
            if layers_list[i] > 0:     
               model.add(Dense(units = layers_list[i], activation='relu', input_dim=n_steps_in))
            save_path = save_path + "["+str(layers_list[i])+"]"
        save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
        # Output layer
        model.add(Dense(units = n_steps_out, activation = 'linear'))
        # Compilling the network according to the loss_metric
        model.compile(optimizer = RMSprop(), loss = correntropy_loss, metrics=[mae_multi, standard_deviation_error, root_mean_square_error])
        # função early stop vai parar de treinar a rede se algum parâmetro monitorado parou de melhorar
        es = EarlyStopping(monitor ='loss', min_delta = 1e-10, patience = 10, verbose = 1)
        # Reduce the learnning rate when the metric stop improving.
        rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.001, patience = 5, verbose = 1)
        mcp =  ModelCheckpoint(filepath=r'./../saves/correntropy/pesos_MLP['+str(fold)+']'+save_path+'_'+label+'_kernel['+str(kernel_size)+'].h5', monitor = 'loss', save_best_only= True)
    
        #training and storing the history
        history = model.fit(x = base_input_training,
                                y= base_output_training, 
                                validation_data = (base_input_test, base_output_test),
                                epochs = 100,
                                batch_size = 500,
                                callbacks = [es,rlr, mcp])
        
        model_json = model.to_json()
        hist = {'loss': str(history.history['loss']),
                'val_loss': str(history.history['val_loss']),
                'mae': str(history.history['mae_multi']),
                'val_mae': str(history.history['val_mae_multi']),
                'rmse': str(history.history['root_mean_square_error']),
                'val_rmse': str(history.history['val_root_mean_square_error']),
                'stddev': str(history.history['standard_deviation_error']),
                'val_stddev': str(history.history['val_standard_deviation_error'])
                }
        j_hist = json.dumps(hist)
        with open(r'./../saves/correntropy/history_MLP['+str(fold)+']'+save_path+'_'+label+'_kernel['+str(kernel_size)+']', 'w') as json_file:
            json_file.write(j_hist)
        with open(r'./../saves/correntropy/regressor_MLP['+str(fold)+']'+save_path+'_'+label+'_kernel['+str(kernel_size)+'].json', 'w') as json_file:
            json_file.write(model_json)


def evaluate_history_corr(label, n_folds, n_steps_in, n_steps_out,layers_list, kernel_size):
    """
    Gets the trainning and Validation data from the history and
    plots the history of training of the MLP. 
    
    Parameters
    ---------- 
    label : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_folds : Integer
        Number of folds for K fold validation 
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict
    layers_list : List
        List containing the number of neurons on each layer.
    kernel_size : float
        kernel_size is a scalar for the kernel size.
    Returns
    -------
    evaluation : dict
        Dictonary containing: loss, val_loss, mae, val_mae,
        rmse, val_rmse, stddev and val_stddev.   
    """
    loss     = []
    val_loss = []
    mae = []
    val_mae = []
    rmse = []
    val_rmse = []
    stddev = []
    val_stddev = []
    
    save_path = ""
    for i in range(len(layers_list)):
        save_path = save_path + "["+str(layers_list[i])+"]"
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    
    for j in range(1,9):
        fold= j
        aux_loss = []
        aux_val_loss = []
        aux_mae = []
        aux_val_mae = []
        aux_rmse = []
        aux_val_rmse = []
        aux_stddev = []
        aux_val_stddev = []
        
        with open(r'./../saves/correntropy/history_MLP['+str(fold)+']'+save_path+'_'+label+'_kernel['+str(kernel_size)+']') as f:
            js = json.load(f)
            aux = js.get("loss")
            aux = aux.split(", ")
            aux_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_loss.append(float(aux[i]))
            aux_loss.append(float(aux[i+1].replace("]", "")))
                
            aux = js.get("val_loss")
            aux = aux.split(", ")
            aux_val_loss.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_loss.append(float(aux[i]))
            aux_val_loss.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("mae")
            aux = aux.split(", ")
            aux_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_mae.append(float(aux[i]))
            aux_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_mae")
            aux = aux.split(", ")
            aux_val_mae.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_mae.append(float(aux[i]))
            aux_val_mae.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("rmse")
            aux = aux.split(", ")
            aux_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_rmse.append(float(aux[i]))
            aux_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("val_rmse")
            aux = aux.split(", ")
            aux_val_rmse.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_rmse.append(float(aux[i]))
            aux_val_rmse.append(float(aux[i+1].replace("]", "")))
            
            aux = js.get("stddev")
            aux = aux.split(", ")
            aux_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_stddev.append(float(aux[i]))
            aux_stddev.append(float(aux[i+1].replace("]", "")))
            
            
            aux = js.get("val_stddev")
            aux = aux.split(", ")
            aux_val_stddev.append(float(aux[0].replace("[", "")))
            for i in range(1, len(aux)-1):
                aux_val_stddev.append(float(aux[i]))
            aux_val_stddev.append(float(aux[i+1].replace("]", "")))

        loss.append(aux_loss)
        val_loss.append(aux_val_loss)
        mae.append(aux_mae)
        val_mae.append(aux_val_mae)
        rmse.append(aux_rmse)
        val_rmse.append(aux_val_rmse)
        stddev.append(aux_stddev)
        val_stddev.append(aux_val_stddev)
    
    
    mean_loss = []
    mean_val_loss = []
    last_mean_loss = []
    last_mean_val_loss = []
    last_mean_mae = []
    last_mean_val_mae = []
    last_mean_rmse = []
    last_mean_val_rmse = []
    last_mean_stddev = []
    last_mean_val_stddev = []
    
    
    for i in range(n_folds):
        last_mean_loss.append(loss[i][len(loss[i])-1])
        last_mean_val_loss.append(val_loss[i][len(loss[i])-1])
        last_mean_mae.append(mae[i][len(mae[i])-1])
        last_mean_val_mae.append(val_mae[i][len(val_mae[i])-1])
        last_mean_rmse.append(rmse[i][len(rmse[i])-1])
        last_mean_val_rmse.append(val_rmse[i][len(val_rmse[i])-1])
        last_mean_stddev.append(stddev[i][len(stddev[i])-1])
        last_mean_val_stddev.append(val_stddev[i][len(val_stddev[i])-1])
    
    last_mean_loss = tf.reduce_mean(last_mean_loss).numpy()
    last_mean_val_loss = tf.reduce_mean(last_mean_val_loss).numpy()
    last_mean_mae = tf.reduce_mean(last_mean_mae).numpy()
    last_mean_val_mae = tf.reduce_mean(last_mean_val_mae).numpy()
    last_mean_rmse = (tf.reduce_mean(last_mean_rmse)).numpy()
    last_mean_val_rmse = (tf.reduce_mean(last_mean_val_rmse)).numpy()
    last_mean_stddev = (tf.reduce_mean(last_mean_stddev)).numpy()
    last_mean_val_stddev = (tf.reduce_mean(last_mean_val_stddev)).numpy()
    
    evaluation = {'loss': last_mean_loss,
                'val_loss': last_mean_val_loss,
                'mae': last_mean_mae,
                'val_mae': last_mean_val_mae,
                'rmse': last_mean_rmse,
                'val_rmse': last_mean_val_rmse,
                'stddev': last_mean_stddev,
                'val_stddev': last_mean_val_stddev}
    
    for i in range(len(loss[0])):
        aux = []
        aux2 = []
        for j in range(n_folds):
            if  i < len(loss[j]):
                aux.append(loss[j][i])
                aux2.append(val_loss[j][i]) 
        mean_loss.append((tf.reduce_mean(aux)).numpy())
        mean_val_loss.append((tf.reduce_mean(aux2)).numpy())
    
    print("Evaluation of trainning and validation:")
    print("loss: ", last_mean_loss, "val_loss: ", last_mean_val_loss,"\n", "mae: ", last_mean_mae, 
          "val_mae: ", last_mean_val_mae, "\n" ,"rmse: ", last_mean_rmse, "val_rmse: ", last_mean_val_rmse,"\n" 
          "stddev: ", last_mean_stddev,"val_stddev: ", last_mean_val_stddev)
    
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(mean_loss, label = "Treinamento")
    ax1.plot(mean_val_loss, label = "Validação")
    ax1.set_ylabel("Média do Erro Quadrático Médio", fontsize = 13)
    ax1.set_xlabel("Épocas", fontsize = 13)
    plt.legend() 
    plt.title("MLP"+save_path+" Correntropy with Kernel size = " + str(kernel_size))
    plt.savefig(r'./../res/mlp/correntropy/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/Plots/MLP'+'_'+label+'_Kernel_'+str(kernel_size)+'.png', dpi=300)
    
    return evaluation

def mlp_univariate_full_corr(label, n_steps_in, n_steps_out, layers_list,kernel_size):
    """
    Using correntropy
    Creates a MLP model to predict the labels and uses all the data for training.
    Saves which history of training in: ../saves/history_MLP'+layer1+layer2+'_'+labels+'.
    Saves which mlp model in: ../saves/regressor_MLP'+layer1+layer2+'_'+labels+'.json'
    
    Parameters
    ----------
    label : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    layers_list : List
        List containing the number of neurons on each layer.
    kernel_size : float
        kernel_size is a scalar for the kernel size.
    Returns
    -------
    Nothing
    """
   
    #Training the model with all the training data
    #Reading the training input data
    input_training = pd.read_csv(r'./../folds/trainingInputData_'+label+'.csv')
    #Getting the column values
    input_training = input_training.values
    #Reading the training output data
    output_training = pd.read_csv(r'./../folds/trainingOutputData_'+label+'.csv')
    output_training = output_training.values
    
    save_path = ""
    #define model
    model = Sequential()
    for i in range(len(layers_list)):
        if layers_list[i] > 0:     
           model.add(Dense(units = layers_list[i], activation='relu', input_dim=n_steps_in))
        save_path = save_path + "["+str(layers_list[i])+"]"
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    # Output layer
    model.add(Dense(units = n_steps_out, activation = 'linear'))
    # Compilling the network with correntropy as loss
    model.compile(optimizer = RMSprop(), loss = correntropy_loss, metrics=[mae_multi, standard_deviation_error, root_mean_square_error])
    es = EarlyStopping(monitor ='loss', min_delta = 1e-10, patience = 10, verbose = 1)
    rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 5, verbose = 1)
    mcp =  ModelCheckpoint(filepath=r'./../saves/correntropy/pesos_MLP'+save_path+'_'+label+'_kernel['+str(kernel_size)+'].h5', monitor = 'loss', save_best_only= True)
    #training and storing the history
    history = model.fit(x = input_training,
                            y= output_training, 
                            epochs = 100,
                            batch_size = 500,
                            callbacks = [es,rlr,mcp])
    model_json = model.to_json()
   
         
    hist = {'loss': str(history.history['loss']),
            'mae': str(history.history['mae_multi']),
            'rmse': str(history.history['root_mean_square_error']),
            'stddev': str(history.history['standard_deviation_error'])
            }
    j_hist = json.dumps(hist)
    with open(r'./../saves/correntropy/history_MLP'+save_path+'_'+label+'_kernel['+str(kernel_size)+']', 'w') as json_file:
        json_file.write(j_hist)
    with open(r'./../saves/correntropy/regressor_MLP'+save_path+'_'+label+'_kernel['+str(kernel_size)+'].json', 'w') as json_file:
       json_file.write(model_json)

def mlp_univariate_test_corr(label, n_steps_in, n_steps_out, layers_list, kernel_size):
    """
    Using correntropy
    Testing the fully trained model with the test data.
    Displays the results in form of: Mean Absolute Error (MAE), 
    Root Mean Square Error (RMS) and Standart Deviation.
    
    Parameters
    ----------
    labels : String
        String containing the  name of the column of the dataset you want to use in your model.
    n_steps_in : Integer
        How many minutes back we use to predict 
    n_steps_out : Integer
        How many minutes forward we want to predict 
    layers_list : List
        List containing the number of neurons on each layer.
    kernel_size : float
        kernel_size is a scalar for the kernel size.
    Returns
    -------
    mae : list
        Mean Absolute Error for each output.
    rms : list
        Root Mean Square for each output.
    stddev : list
        Standart Deviation for each output.
    """
    #Testing the model fully trained
    #Reading the test input data
    input_test = pd.read_csv(r'./../folds/testInputData_'+label+'.csv')
    #Getting the column values
    input_test = input_test.values
    #Reading the training output data
    output_test = pd.read_csv(r'./../folds/testOutputData_'+label+'.csv')
    output_test = output_test.values
    
    save_path = ""
    for i in range(len(layers_list)):
        save_path = save_path + "["+str(layers_list[i])+"]"
    save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
    # Openning the file which contains the network model
    mlp_file = open(r'./../saves/correntropy/regressor_MLP'+save_path+'_'+label+'_kernel['+str(kernel_size)+'].json', 'r')
    mlp_structure = mlp_file.read()
    # Fechando o arquivo
    mlp_file.close()
    # Getting the network structure
    model = model_from_json(mlp_structure)
    # Reading the weights the putting them  in the network model
    model.load_weights(r'./../saves/correntropy/pesos_MLP'+save_path+'_'+label+'_kernel['+str(kernel_size)+'].h5')
        
    predictions = model.predict(input_test)

    normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')
    
    y = normalizator.inverse_transform(output_test)
    y_hat = normalizator.inverse_transform(predictions)
    
    mae, rmse, stddev = mae_rmse_stddev_evaluation(y, y_hat)
    
    print("AVARAGE "+ label +" prediction:")
    for i in range(len(mae)):
        print(str(i+1)+" Output minute:")
        print("Mean Absolute Error (MAE): {:.3f}".format(mae[i]))
        print("Root Mean Square Error (RMS): {:.3f}".format(rmse[i]))
        print("Standart Deviation: {:.3f}".format(stddev[i]))
    
    return mae, rmse, stddev
    

