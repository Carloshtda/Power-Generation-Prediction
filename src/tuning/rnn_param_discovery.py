from utils import *
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime

#%%
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 0, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32), activation='relu'))
    model.add(layers.Dense(5, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='mean_absolute_error', 
        metrics=['mean_absolute_error'])
    return model

# log_dir = "logs/" + datetime.now().strftime("%m%d-%H%M")
# log_dir = "logs/0408-0122"

# training meta
es = EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 5)

# tb = tf.keras.callbacks.TensorBoard(
#     log_dir=log_dir,
#     histogram_freq=1,
#     embeddings_freq=1,
#     write_graph=True,
#     update_freq='batch')


#%%
tuner = BayesianOptimization(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=10,
    executions_per_trial=3,
    directory='.',
    project_name='mlp_BO_param',
    overwrite=True)
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
input_training = input_training.reshape(input_training.shape[0], input_training.shape[1],1)
input_test = input_test.reshape(input_test.shape[0], input_test.shape[1], 1)
#%%
tuner.search(input_training, output_training, epochs = 128, validation_data=(input_test, output_test), callbacks=[es])
#%%
tuner.results_summary()
#%%
hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
