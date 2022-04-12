#%%
from utils import *
    
#%%
if __name__ == '__main__':
    folder = os.path.join(".","..","..","db","Dados Sistema 5 kW","Ano 2020")
    
    input_labels = ["Potencia_FV_Avg"]
    output_labels = ["Potencia_FV_Avg"]
    n_steps_in = 120
    n_steps_out = 5
    
    #generate_train_test_valid(folder, input_labels, output_labels, n_steps_in, n_steps_out)
    generate_train_test_valid_time(folder, input_labels, output_labels, n_steps_in, n_steps_out)


# %%
