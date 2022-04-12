# %%
import os
import json
from tracemalloc import start
import numpy as np
import pandas as pd
import datetime
from matplotlib import dates
from utils import *
#%%
def baseline_mean(folder, label, time_shift, amount):
    lst = get_list_of_files(folder)
    dfs = (read_dat_file(f) for f in lst)
    df_complete = pd.concat(dfs, ignore_index=True)
    
    df_label = df_complete[label]
    df_label = df_label.dropna()
    df_label = df_label.reset_index(drop=True)
    indexAux = df_label[(df_label < 0)].index
    df_label[indexAux] = 0.0
    
    y = []
    y_hat = []
    for i in range(time_shift*amount, np.size(df_label)-1):
        y.append(df_label[i])
        aux = []
        for day in range(1, amount+1):
            aux.append(df_label[i-(time_shift*day)])
        y_hat.append(np.array(aux).mean())
    
    return y, y_hat

    
#%%
def disp_plot(y_true,y_pred, save_path):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.scatter(y_true, y_pred, s=1, c='b')
    ax1.plot(y_true, y_true, color = 'r')
    ax1.set_ylabel("Real", fontsize = 13)
    ax1.set_xlabel("Previsto", fontsize = 13)
    plt.title("Gráfico de Dispersão - Baseline", fontsize = 13)
    plt.savefig(save_path, dpi=120)
        
    return
def pred_plot(y_true,y_pred, save_path):    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    t = np.arange(0+360, len(y_pred)-360, 1)
    times=np.array([datetime.datetime(2019, 9, 27, int(p/60), int(p%60), int(0)) for p in t])
    fmtr = dates.DateFormatter("%H:%M")


    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(times,y_true[360:len(y_pred)-360],linestyle='-',color= 'red',label = 'Real', linewidth=1.5)
    ax1.plot(times,y_pred[360:len(y_pred)-360],linestyle='--', color= 'royalblue', label = 'Predicted', linewidth=2,dashes=(1, 2))
    ax1.xaxis.set_major_formatter(fmtr)
    
    ax1.tick_params(axis='x', labelsize= 18)
    ax1.tick_params(axis='y', labelsize= 18)
    
    ax1.set_ylabel("Power (W)", fontsize = 20)
    ax1.set_xlabel("Hour", fontsize = 20)
    #plt.title("Gráfico Real x Predito - Minuto "+str(minute), fontsize = 18)
    plt.legend(fontsize = 'small')
    plt.grid(b=True)
    #plt.savefig(save_path, dpi=300)
    plt.savefig(save_path,  dpi=1200)
    
    return
# %%
folder = os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020")
lst = get_list_of_files(folder)
dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)
# %%
#1. A previsão equivale ao valor do minuto anterior
time_shift = 1440
output_labels = ["Potencia_FV_Avg"]
days_amount = 10

y, y_hat = baseline_mean(folder, output_labels[0], time_shift, days_amount)

# %%
mae, rms, stddev = mae_rmse_stddev_evaluation(y, y_hat)
    
print("AVARAGE baseline prediction:")
print("Mean Absolute Error (MAE): {:.3f}".format(mae))
print("Root Mean Square Error (RMS): {:.3f}".format(rms))
print("Standart Deviation: {:.3f}".format(stddev))
# %%
save_path = './../temp/disp'
disp_plot(y, y_hat, save_path)
# %%
init = 470753
shift = 420
save_path = './../temp/pred'
pred_plot(y[init+shift:init+1440+shift], y_hat[init+shift:init+1440+shift], save_path)
# %%
