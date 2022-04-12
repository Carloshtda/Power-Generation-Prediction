#%% Import modules
from utils import *
#%%
folder = os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020")
lst = get_list_of_files(folder)
dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)
# radiacao_df = df_complete.pop('Potencia_FV_Avg')
# radiacao_tf = tf.data.Dataset.from_tensor_slices(radiacao_df.values)
#%%
#Preprocessing, Baseline and MLP starts here
input_labels = ["Potencia_FV_Avg"]
output_labels = ["Potencia_FV_Avg"]
n_steps_in = 120
n_steps_out = 5
n_folds = 8
layers_list = [96,96,224,32]
day_only = False
net_type ="mlp"
loss_metric ="mse"
#%%

df_label = df_complete[output_labels[0]]
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)
#Negative numbers turn into 0 for each label
indexAux = df_label[(df_label < 0)].index
df_label[indexAux] = 0.0

time_shift = 1440 
#List with the real and predicted (shifted 24h hours back) valeus of radiation
mae =[]
ind = []
aux =0
# for i in range(470753+120+time_shift, np.size(df_label)-1-3):
for i in range(470753+time_shift, np.size(df_label)-1-3):
    y = df_label[i]
    y_hat = df_label[i-time_shift]
    if y >= 1:
        ind.append(aux)  
        mae.append(backend.mean(backend.abs(tf.math.subtract(y , y_hat))).numpy())
    aux = aux +1
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

predictions = model.predict(input_test)
normalizator = joblib.load(r'./../saves/norm/normPotencia_FV_Avg.save')

y = normalizator.inverse_transform(output_test)
y_hat = normalizator.inverse_transform(predictions)    
mae_mlp = []
aux =0
for i in range(470753+time_shift, np.size(df_label)-1-3):
    if aux in ind:
        mae_mlp.append(backend.mean(backend.abs(tf.math.subtract(y[aux] , y_hat[aux]))).numpy())
    aux = aux + 1
    

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from pylab import rcParams
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

#%%
rcParams['figure.figsize'] = 20,10
rcParams['font.size'] = 30
sns.set()
np.random.seed(8)


def plot_distribution(inp):
    plt.figure()
    ax = sns.distplot(inp)
    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
    _, max_ = plt.ylim()
    plt.text(
        inp.mean() + inp.mean() / 10,
        max_ - max_ / 10,
        "Mean: {:.2f}".format(inp.mean()),
    )
    return plt.figure

#%%
mae = np.array(mae)
plot_distribution(mae)
#%%
mae_mlp = np.array(mae_mlp)
plot_distribution(mae_mlp)

#%%


plt.figure()
ax1 = sns.distplot(mae)
ax2 = sns.distplot(mae_mlp)
plt.axvline(np.mean(mae), color='b', linestyle='dashed', linewidth=5)
plt.axvline(np.mean(mae_mlp), color='orange', linestyle='dashed', linewidth=5)

#%%
def compare_2_groups(arr_1, arr_2, alpha, sample_size):
    stat, p = ttest_ind(arr_1, arr_2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
#%%
sample_size = 1000
base_sampled = np.random.choice(mae, sample_size)
mlp_sampled = np.random.choice(mae_mlp, sample_size)
compare_2_groups(base_sampled, mlp_sampled, 0.05, sample_size)

#%%
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p


# generate two independent samples
data1 = mae
data2 = mae_mlp
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')


#%%
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import t

# function for calculating the t-test for two dependent samples
def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

# generate two dependent samples
data1 = mae
data2 = mae_mlp
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = dependent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')




























#%%
from scipy.stats import ttest_rel
#%%
t_test = ttest_rel(mae, mae_mlp)
print(t_test)
t, p = ttest_rel(mae, mae_mlp)
print(t)
print(p)
#%%
import pingouin as pt
#%% 
t_test = pt.ttest(mae,mae_mlp, paired=True)
#%% 
t_test