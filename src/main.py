#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:24:39 2019

@author: tvieira
"""
# Download database:
# https://drive.google.com/drive/folders/1pljxBfGNXp3OvRA4Czos6Uufnx9uhtZ3?usp=sharing

#%% Import modules
from utils import *
#%%
folder = os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020")
#lst = get_list_of_files(folder)
#dfs = (read_dat_file(f) for f in lst)
#df_complete = pd.concat(dfs, ignore_index=True)
# radiacao_df = df_complete.pop('Potencia_FV_Avg')
# radiacao_tf = tf.data.Dataset.from_tensor_slices(radiacao_df.values)
#%%
#Preprocessing, Baseline and MLP starts here
input_labels = ["Potencia_FV_Avg"]
output_labels = ["Potencia_FV_Avg"]
n_steps_in = 120
n_steps_out = 5
n_folds = 8
#layers_list = [32, 32, 256, 32, 256, 16]
layers_list = [5]
day_only = False
net_type ="lstm"
loss_metric ="mse"
#kernel_size = 0.1

#%%
#Baseline
time_shift = 1440 #Equivalent to 24h
base_mae, base_rmse, base_stddev = generate_baseline(folder, output_labels[0], time_shift, day_only)


#%%
#Prepocessing the data and dividing it into trainning, validation, and test datasets  
generate_train_test_valid(folder, input_labels, output_labels, n_steps_in, n_steps_out, n_folds, day_only)

#%%
#Training and Testing
init_gpus()
if net_type == 'mlp':
    if loss_metric == "correntropy":
        mlp_univariate_kfold_corr(input_labels, n_steps_in, n_steps_out,n_folds, layers_list, kernel_size)
        eval_hist = evaluate_history_corr(input_labels, n_folds, n_steps_in, n_steps_out,layers_list, kernel_size)
        mlp_univariate_full_corr(input_labels, n_steps_in, n_steps_out, layers_list,kernel_size)
        mae, rmse, stddev = mlp_univariate_test_corr(input_labels, n_steps_in, n_steps_out, layers_list, kernel_size)
        end_path = '_kernel_'+'['+str(kernel_size)+']'
        
    else:
        mlp_kfold(n_steps_in,n_steps_out, input_labels, n_folds, layers_list)
        eval_hist = evaluate_mlp_history(n_steps_in, n_steps_out, input_labels, n_folds, layers_list)
        mlp_univariate_full(n_steps_in, n_steps_out, input_labels, layers_list)
        mae, rmse, mlp_stddev = mlp_univariate_test(n_steps_in, n_steps_out, input_labels, layers_list)
        end_path = '_daytime_'+str(day_only)+''
elif net_type == 'rnn':
    rnn_kfold(n_steps_in,n_steps_out, input_labels, n_folds, layers_list)
    eval_hist = evaluate_rnn_history(n_steps_in, n_steps_out, input_labels, n_folds, layers_list)
    rnn_full(n_steps_in, n_steps_out, input_labels, layers_list)
    mae, rmse, stddev = rnn_test(n_steps_in, n_steps_out, input_labels, layers_list)
else:
    lstm_kfold(n_steps_in,n_steps_out, input_labels, n_folds, layers_list)
    eval_hist = evaluate_lstm_history(n_steps_in, n_steps_out, input_labels, n_folds, layers_list)
    lstm_full(n_steps_in, n_steps_out, input_labels, layers_list)
    mae, rmse, stddev = lstm_test(n_steps_in, n_steps_out, input_labels, layers_list)
#%%    
#Writting the results in CSV  

save_path = ""
for i in range(len(layers_list)):
    save_path = save_path + "["+str(layers_list[i])+"]"
for i in range(len(input_labels)):
    save_path = save_path +"["+str(input_labels[i]) + "]"
if net_type == 'mlp':
    f = open(r'./../res/mlp/'+loss_metric+'/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/MLP'+save_path+end_path+'.csv', 'w')
    with f:
        
        fnames = ['Model','Mean Absolute Error (MAE)', 'Root Mean Square (RMS)', 'Standart Deviation']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()
        writer.writerow({'Model' : "Baseline", 'Mean Absolute Error (MAE)' : base_mae, 'Root Mean Square (RMS)': base_rmse, 'Standart Deviation' : base_stddev})
        writer.writerow({'Model' : "Trainning", 'Mean Absolute Error (MAE)' : eval_hist.get("mae"), 'Root Mean Square (RMS)': eval_hist.get("rmse"), 'Standart Deviation' : eval_hist.get("stddev")})
        writer.writerow({'Model' : "Validation", 'Mean Absolute Error (MAE)' : eval_hist.get("val_mae"), 'Root Mean Square (RMS)': eval_hist.get("val_rmse"), 'Standart Deviation' : eval_hist.get("val_stddev")})
        for i in range(len(mae)):
            writer.writerow({'Model' : "Test Output "+str(i+1) , 'Mean Absolute Error (MAE)' : mae[i], 'Root Mean Square (RMS)': rmse[i], 'Standart Deviation' : stddev[i]})
        writer.writerow({'Model' : "Test Output Mean" , 'Mean Absolute Error (MAE)' : np.mean(mae), 'Root Mean Square (RMS)':  np.mean(rmse), 'Standart Deviation' : np.mean(stddev)})
elif net_type == 'lstm':
    normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')
    f = open(r'./../res/lstm/'+loss_metric+'/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/RNN'+save_path+'.csv', 'w')
    with f:
    
        fnames = ['Model','Mean Absolute Error (MAE)', 'Root Mean Square (RMS)', 'Standart Deviation']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()
        writer.writerow({'Model' : "Baseline", 'Mean Absolute Error (MAE)' : base_mae, 'Root Mean Square (RMS)': base_rmse, 'Standart Deviation' : base_stddev})
        writer.writerow({'Model' : "Trainning", 'Mean Absolute Error (MAE)' : normalizator.inverse_transform(np.array(float(eval_hist.get("mae"))).reshape(-1, 1)), 'Root Mean Square (RMS)': normalizator.inverse_transform(np.array(float(eval_hist.get("rmse"))).reshape(-1, 1)), 'Standart Deviation' : normalizator.inverse_transform(np.array(float(eval_hist.get("stddev"))).reshape(-1, 1))})
        writer.writerow({'Model' : "Validation", 'Mean Absolute Error (MAE)' : normalizator.inverse_transform(np.array(float(eval_hist.get("val_mae"))).reshape(-1, 1)), 'Root Mean Square (RMS)': normalizator.inverse_transform(np.array(float(eval_hist.get("val_rmse"))).reshape(-1, 1)), 'Standart Deviation' :normalizator.inverse_transform(np.array(float(eval_hist.get("val_stddev"))).reshape(-1, 1))})
        for i in range(len(mae)):
            writer.writerow({'Model' : "Test Output "+str(i+1) , 'Mean Absolute Error (MAE)' : mae[i], 'Root Mean Square (RMS)': rmse[i], 'Standart Deviation' :  stddev[i]})
        writer.writerow({'Model' : "Test Output Mean" , 'Mean Absolute Error (MAE)' : np.mean(mae), 'Root Mean Square (RMS)':  np.mean(rmse), 'Standart Deviation' : np.mean(stddev)})
else:
    normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')
    f = open(r'./../res/rnn/'+loss_metric+'/'+str(n_steps_in)+'minBack_'+str(n_steps_out)+'minForward/RNN'+save_path+'.csv', 'w')
    with f:
    
        fnames = ['Model','Mean Absolute Error (MAE)', 'Root Mean Square (RMS)', 'Standart Deviation']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()
        writer.writerow({'Model' : "Baseline", 'Mean Absolute Error (MAE)' : base_mae, 'Root Mean Square (RMS)': base_rmse, 'Standart Deviation' : base_stddev})
        writer.writerow({'Model' : "Trainning", 'Mean Absolute Error (MAE)' : normalizator.inverse_transform(np.array(float(eval_hist.get("mae"))).reshape(-1, 1)), 'Root Mean Square (RMS)': normalizator.inverse_transform(np.array(float(eval_hist.get("rmse"))).reshape(-1, 1)), 'Standart Deviation' : normalizator.inverse_transform(np.array(float(eval_hist.get("stddev"))).reshape(-1, 1))})
        writer.writerow({'Model' : "Validation", 'Mean Absolute Error (MAE)' : normalizator.inverse_transform(np.array(float(eval_hist.get("val_mae"))).reshape(-1, 1)), 'Root Mean Square (RMS)': normalizator.inverse_transform(np.array(float(eval_hist.get("val_rmse"))).reshape(-1, 1)), 'Standart Deviation' :normalizator.inverse_transform(np.array(float(eval_hist.get("val_stddev"))).reshape(-1, 1))})
        for i in range(len(mae)):
            writer.writerow({'Model' : "Test Output "+str(i+1) , 'Mean Absolute Error (MAE)' : mae[i], 'Root Mean Square (RMS)': rmse[i], 'Standart Deviation' :  stddev[i]})
        writer.writerow({'Model' : "Test Output Mean" , 'Mean Absolute Error (MAE)' : np.mean(mae), 'Root Mean Square (RMS)':  np.mean(rmse), 'Standart Deviation' : np.mean(stddev)})