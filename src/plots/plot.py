#%% Import modules
from utils import *
import datetime
from matplotlib import dates
#%%
input_labels = ["Potencia_FV_Avg"]
output_labels = ["Potencia_FV_Avg"]
n_steps_in = 120
n_steps_out = 5
n_folds = 8
layers_list = [96,96,224,32]
day_only = False
net_type ="lstm"
loss_metric ="mse"

#%%
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
#%%
shift = 300


predictions = model.predict(input_test[0+shift:1440+shift])

normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')

y = normalizator.inverse_transform(output_test[0+shift:1440+shift])
y_hat = normalizator.inverse_transform(predictions)    

y_true = np.array(y)
y_pred = np.array(y_hat)
#%%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 22})
#%%
t = np.arange(0+360, len(predictions)-360, 1)
times=np.array([datetime.datetime(2019, 9, 27, int(p/60), int(p%60), int(0)) for p in t])
fmtr = dates.DateFormatter("%H:%M")
#%%
plot_size = y_true.shape[1]
save_path = "D:\\EngComp\\Pesquisa\\Preditor de potencia\\energy\\img\\PM"
for i in range(plot_size):
    minute = i + 1
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(times,y_true[360:len(predictions)-360,i],linestyle='-',color= 'red',label = 'Real', linewidth=1.5)
    ax1.plot(times,y_pred[360:len(predictions)-360,i],linestyle='--', color= 'royalblue', label = 'Predicted', linewidth=2,dashes=(1, 2))
    ax1.xaxis.set_major_formatter(fmtr)
    
    ax1.tick_params(axis='x', labelsize= 18)
    ax1.tick_params(axis='y', labelsize= 18)
    
    ax1.set_ylabel("Power (W)", fontsize = 20)
    ax1.set_xlabel("Hour", fontsize = 20)
    #plt.title("Gráfico Real x Predito - Minuto "+str(minute), fontsize = 18)
    plt.legend(fontsize = 'small')
    plt.grid(b=True)
    #plt.savefig(save_path, dpi=300)
    plt.savefig(save_path+str(i+1)+'.png', format='png',  dpi=1200)
#%%
predictions = model.predict(input_test)
y = normalizator.inverse_transform(output_test)
y_hat = normalizator.inverse_transform(predictions)    
y_true = np.array(y)
y_pred = np.array(y_hat)

save_path = "D:\\EngComp\\Pesquisa\\Preditor de potencia\\energy\\img\\Dispersao_"
plot_size = y_true.shape[1]
for i in range(plot_size):
    minute = i + 1 
    fig = plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.scatter(y_true[:,i], y_pred[:,i], s=1, c='b')
    ax1.plot(y_true[:,i], y_true[:,i], color = 'r')
    
    ax1.tick_params(axis='x', labelsize= 18)
    ax1.tick_params(axis='y', labelsize= 18)
    
    ax1.set_ylabel("Real Power (W)", fontsize = 20)
    ax1.set_xlabel("Predicted Power (W)", fontsize = 20)
    plt.grid(b=True)

    plt.savefig(save_path+str(i+1)+'.png', format='png',  dpi=1200)