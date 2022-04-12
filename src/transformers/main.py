#%%
from utils import *
from model import build_model
#%%
init_gpus()

#x_train = pd.read_csv(r'./db/data/trainingInputData.csv')
x_train = pd.read_csv(r'./../../db/data/trainingInputDataTime.csv')
x_train = x_train.values
#x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_train = x_train.reshape((x_train.shape[0], int(x_train.shape[1]/5), 5))

#y_train = pd.read_csv(r'./db/data/trainingOutputData.csv')
y_train = pd.read_csv(r'./../../db/data/trainingOutputDataTime.csv')
y_train = y_train.values

#x_test = pd.read_csv(r'./db/data/testInputData.csv')
x_test = pd.read_csv(r'./../../db/data/testInputDataTime.csv')
x_test = x_test.values
#x_test = x_test.reshape((x_test.shape[0], int(x_test.shape[1]), 1))
x_test = x_test.reshape((x_test.shape[0], int(x_test.shape[1]/5), 5))

#y_test = pd.read_csv(r'./db/data/testOutputData.csv')
y_test = pd.read_csv(r'./../../db/data/testOutputDataTime.csv')
y_test = y_test.values


input_shape = x_train.shape[1:]
output_len = y_train.shape[1]

#%%
model = build_model(
    input_shape=input_shape,
    output_len=output_len,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=8,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)
#%%
# Compilling the network according to the loss_metric
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])
# função early stop vai parar de treinar a rede se algum parâmetro monitorado parou de melhorar
es = keras.callbacks.EarlyStopping(monitor ='val_loss', patience = 10, restore_best_weights=True)
# Reduce the learnning rate when the metric stop improving.
rlr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, verbose = 1)
mcp =  keras.callbacks.ModelCheckpoint(filepath=r'./../../db/saves/pesos_Transformersh8.h5', monitor = 'val_loss', save_best_only= True)

model.summary()
#%%
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=[es, rlr, mcp],
)
#%%
model_json = model.to_json()
hist_dict = {'loss': str(history.history['loss'][-1]),
        'val_loss': str(history.history['val_loss'][-1]),
        'mae': str(history.history['mae_multi'][-1]),
        'val_mae': str(history.history['val_mae_multi'][-1]),
        'rmse': str(history.history['root_mean_square_error'][-1]),
        'val_rmse': str(history.history['val_root_mean_square_error'][-1]),
        'stddev': str(history.history['standard_deviation_error'][-1]),
        'val_stddev': str(history.history['val_standard_deviation_error'][-1])
        }
j_hist = json.dumps(hist_dict)

with open(r'./../../db/saves/history_Transformersh8', 'w') as json_file:
    json_file.write(j_hist)
with open(r'./../../db/saves/regressor_Transformersh8.json', 'w') as json_file:
    json_file.write(model_json)
#%%
eva = model.evaluate(x_test, y_test, verbose=1)

#%%
