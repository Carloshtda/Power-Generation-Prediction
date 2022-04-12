#%% Import modules
from utils import *


#%%
x = ['1º','2º','3º','4º','5º']
mae = [68.371,94.983, 108.708, 118.009, 125.315]
std = [224.352, 283.220, 310.755, 326.753, 339.339]

#%%
save_path = "D:\\EngComp\\Pesquisa\\Preditor de potencia\\energy\\img\\minute.png"

fig = plt.figure(figsize=(8,6))
ax1=fig.add_subplot(1, 1, 1)
ax1.plot(x, mae,linestyle='-',color= 'red',label = 'MAE', linewidth=2, marker = "s")
ax1.plot(x, std,linestyle='-', color= 'royalblue', label = 'STD', linewidth=2, marker = "s")

ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)

ax1.set_ylabel("Watts (W)", fontsize = 20)
ax1.set_xlabel("Minute", fontsize = 20)

plt.legend(loc='best', fontsize = 'small')
ax1.set_ylim([0, 500])
plt.grid(b=True)
plt.savefig(save_path, format='png',  dpi=1200)