from utils import *
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
n_input = 6
#%%
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 0, 5)):
        model.add(LSTM(units=hp.Int('units',min_value=32,
                                        max_value=512,
                                        step=32), 
                   activation='tanh', input_shape=(n_input, 1)))
    model.add(LSTM(units=hp.Int('units_' + str(i), min_value=16, max_value=64, step=16), activation='tanh'))
    model.add(Dense(5), activation='linear')
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model
#%%
bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=3,
    executions_per_trial=1,
    directory=os.path.normpath('C:/keras_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)
#%%
bayesian_opt_tuner.search(train_x, train_y,epochs=n_epochs,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)

#%%
bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]