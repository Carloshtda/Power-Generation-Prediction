#%%
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './../')


from utils import *
from tensorflow import keras
import keras_tuner as kt
n_input = 6
#%%
input_training = pd.read_csv(r'./../../folds/trainingInputData.csv')
input_training = input_training.values

output_training = pd.read_csv(r'./../../folds/trainingOutputData.csv')
output_training = output_training.values

input_test = pd.read_csv(r'./../../folds/testInputData.csv')
input_test = input_test.values

output_test = pd.read_csv(r'./../../folds/testOutputData.csv')
output_test = output_test.values

input_training = input_training.reshape(input_training.shape[0], input_training.shape[1],1)
input_test = input_test.reshape(input_test.shape[0], input_test.shape[1], 1)

#%%
def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=256,step=32),
            return_sequences=True, 
            input_shape=(input_training.shape[1],input_training.shape[2]),
            activation='tanh'))
    for i in range(hp.Int('n_layers', 0, 3)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=256,step=32),return_sequences=True, activation='tanh'))
    model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=256,step=32), activation='tanh'))
    model.add(Dense(output_training.shape[1], activation='linear'))
    model.compile(loss='mse', metrics=['mse'], 
                optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])))
    return model


#%%
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 0, 5)):
        model.add(LSTM(units=hp.Int('units',min_value=32,
                                        max_value=512,
                                        step=32), 
                   activation='tanh', input_shape=(n_input, 1)))
    model.add(LSTM(units=hp.Int('units', min_value=16, max_value=64, step=16), activation='tanh'))
    model.add(Dense(5, activation='linear'))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model
#%%
bayesian_opt_tuner = kt.BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=3,
    executions_per_trial=1,
    directory=os.path.normpath('lstm_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

#%%
bayesian_opt_tuner.search(input_training, output_training,epochs=128,
     #validation_data=(X_test, y_test)
     validation_split=0.2, verbose=1)

#%%
bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]