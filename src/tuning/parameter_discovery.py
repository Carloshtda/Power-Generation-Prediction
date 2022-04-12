#%%
from utils import *
from tensorflow import keras
import keras_tuner as kt

#%%
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 1, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=256,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(5, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model
tuner = kt.RandomSearch(build_model, objective='val_mean_absolute_error', max_trials=5, executions_per_trial=3,
                     directory='./tuning/',project_name='mlpP_param')
#%%
tuner.search_space_summary()
#%%
#Reading the training and testing data
input_training = pd.read_csv(r'./../folds/trainingInputData.csv')
input_training = input_training.values
output_training = pd.read_csv(r'./../folds/trainingOutputData.csv')
output_training = output_training.values
input_test = pd.read_csv(r'./../folds/testInputData.csv')
input_test = input_test.values
output_test = pd.read_csv(r'./../folds/testOutputData.csv')
output_test = output_test.values
#%%
tuner.search(input_training, output_training, epochs = 128, validation_data=(input_test, output_test))
#%%
tuner.results_summary()
#%%
hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
scr = tuner.oracle.get_best_trials(num_trials=1)[0].score


