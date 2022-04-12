#%% Import modules
from utils import *
#%%
folder = os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020")
lst = get_list_of_files(folder)
dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)
#radiacao_df = df_complete.pop('Potencia_FV_Avg')
# radiacao_tf = tf.data.Dataset.from_tensor_slices(radiacao_df.values)
#%%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 22})

#%%
fig = plt.figure(figsize=(8,6))
ax1=fig.add_subplot(1, 1, 1)
plt.boxplot(df_complete['Potencia_FV_Avg'])
plt.xticks([1], [''])
ax1.set_ylabel("Power (W)", fontsize = 20)
ax1.set_xlabel("Average PV Power", fontsize = 20)
ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)


plt.grid(b=True)
plt.savefig('D:\EngComp\Pesquisa\Preditor de potencia\energy\img\\boxplot.png', dpi=600)