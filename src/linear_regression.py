# %%
from utils import *
from sklearn.linear_model import LinearRegression

#%%
folder = os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020")

input_label = "Potencia_FV_Avg"
n_steps_in = 120
n_steps_out = 5

#%%
lst = get_list_of_files(folder)
dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)
#%%
df_label = df_complete[[input_label]]
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)
#%%
df_label['Time'] = np.arange(len(df_label.index))

'''
for i in range(1,n_steps_in):
    df_label[f'Lag_{i}'] = df_label[input_label].shift(i)
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)
'''
#%%
#Splitting the data into training and test data.
trainingData, testData =  train_test_split(df_label,test_size = 0.1, shuffle=False)
trainingData = trainingData.reset_index(drop=True)
testData = testData.reset_index(drop=True)
#%%
#Splitting the test data into input and output.
training_input, training_output = split_sequence(trainingData, n_steps_in, n_steps_out, [input_label], [input_label])
test_input, test_output = split_sequence(testData, n_steps_in, n_steps_out, [input_label], [input_label])
# Flatten the shape of the input and output samples. 
training_input = training_input.reshape(training_input.shape[0], training_input.shape[1]*training_input.shape[2])
training_output = training_output.reshape(training_output.shape[0], training_output.shape[1]*training_output.shape[2])

test_input = test_input.reshape(test_input.shape[0], test_input.shape[1]*test_input.shape[2])
test_output = test_output.reshape(test_output.shape[0], test_output.shape[1]*test_output.shape[2])
# %%
model = LinearRegression()
model.fit(training_input, training_output)
#%%
y_hat = model.predict(test_input)
# %%
mae, rmse, stddev = mae_rmse_stddev_evaluation(test_output, y_hat)
# %%
print("Tests evaluation per minute:")
for i in range(len(mae)):
    print(str(i+1)+" Output minute:")
    print("Mean Absolute Error (MAE): {:.3f}".format(mae[i]))
    print("Root Mean Square Error (RMS): {:.3f}".format(rmse[i]))
    print("Standart Deviation: {:.3f}".format(stddev[i]))
# %%
print("Tests evaluation in total:")
print("Mean Absolute Error (MAE): {:.3f}".format(np.mean(mae)))
print("Root Mean Square Error (RMS): {:.3f}".format(np.mean(rmse)))
print("Standart Deviation: {:.3f}".format(np.mean(stddev)))
# %%
