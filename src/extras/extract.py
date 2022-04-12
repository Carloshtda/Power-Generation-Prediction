#%% Import modules
from utils import *
#%% 
input_labels = ["Potencia_FV_Avg"]
output_labels = ["Potencia_FV_Avg"]
layers_list = [60]
n_steps_in = 120
n_steps_out = 5
save_path = ""
for i in range(len(layers_list)):
    save_path = save_path + "["+str(layers_list[i])+"]"
save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
for i in range(len(input_labels)):
    save_path = save_path + "["+str(input_labels[i]) + "]" 

mlp_file = open(r'./../saves/mse/regressor_MLP'+save_path+'.json', 'r')
mlp_structure = mlp_file.read()
mlp_file.close()
model = model_from_json(mlp_structure)
model.load_weights(r'./../saves/mse/pesos_MLP'+save_path+'.h5')
print("MLP: ")
model.summary()
#%% 
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
print("RNN: ")
model.summary()
#%%
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
print("LSTM: ")
model.summary()