#%%
from utils import *
#%%
def dispersion_plot(y_true,y_pred, save_path):
    """
    Plot and saves a dispersion plot between the real value and the predicted value.
    
    Parameters
    ----------
    y_true : Series
        Containing the real value.
    y_pred : Integer
        Containing the predict value.  
    save_path : String
        String containing the path to save the model.
    Returns
    -------
   
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    plot_size = y_true.shape[1]
    for i in range(plot_size):
        minute = i + 1 
        fig = plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(1, 1, 1)
        ax1.scatter(y_true[:,i], y_pred[:,i], s=1, c='b')
        ax1.plot(y_true[:,i], y_true[:,i], color = 'r')
        ax1.set_ylabel("Real", fontsize = 13)
        ax1.set_xlabel("Previsto", fontsize = 13)
        plt.title("Gráfico de Dispersão (w/time)- Minuto "+str(minute), fontsize = 13)
        aux_path = save_path + f"dispersion minute {minute}(time).png"
        plt.savefig(aux_path, dpi=120)
        
    return

def predictions_plot(y_true,y_pred, save_path):
    """
    Plot and saves plot the real value and the predicted value x Timesteps.
    
    Parameters
    ----------
    y_true : Series
        Containing the real value.
    y_pred : Integer
        Containing the predict value. 

    save_path : String
        String containing the path to save the model.
    Returns
    -------
   
    """    
    shift = 300

    y_true = np.array(y_true[0+shift:1440+shift])
    y_pred = np.array(y_pred[0+shift:1440+shift])

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 22})

    t = np.arange(0+360, len(y_pred)-360, 1)
    times=np.array([datetime.datetime(2019, 9, 27, int(p/60), int(p%60), int(0)) for p in t])
    fmtr = dates.DateFormatter("%H:%M")

    plot_size = y_true.shape[1]
    for i in range(plot_size):
        minute = i + 1
        fig = plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(1, 1, 1)
        ax1.plot(times,y_true[360:len(y_pred)-360,i],linestyle='-',color= 'red',label = 'Real', linewidth=1.5)
        ax1.plot(times,y_pred[360:len(y_pred)-360,i],linestyle='--', color= 'royalblue', label = 'Predicted', linewidth=2,dashes=(1, 2))
        ax1.xaxis.set_major_formatter(fmtr)
        
        ax1.tick_params(axis='x', labelsize= 18)
        ax1.tick_params(axis='y', labelsize= 18)
        
        ax1.set_ylabel("Power (W)", fontsize = 20)
        ax1.set_xlabel("Hour", fontsize = 20)
        plt.title("Gráfico Real x Predito(w/time) - Minuto "+str(minute), fontsize = 18)
        plt.legend(fontsize = 'small')
        plt.grid(b=True)
        aux_path = save_path + f"prediction minute {minute}(time).png"
        plt.savefig(aux_path, dpi=1200)
    return
#%%
if __name__ == '__main__':
    transf_file = open(r'./../../db/saves/regressor_Transformers_time.json', 'r')
    transf_structure = transf_file.read()
    transf_file.close()
    # Getting the network structure
    model = keras.models.model_from_json(transf_structure)
    # Reading the weights the putting them  in the network model
    model.load_weights(r'./../../db/saves/pesos_Transformers_time.h5')

    x_test = pd.read_csv(r'./../../db/data/testInputDataTime.csv')
    x_test = x_test.values
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_test = x_test.reshape(x_test.shape[0], int(x_test.shape[1]/5), 5)

    y_test = pd.read_csv(r'./../../db/data/testOutputDataTime.csv')
    y_test = y_test.values

    predictions = model.predict(x_test)

    plot_save_path = r'./../../db/plots/dispersion/'
    dispersion_plot(y_test, predictions, plot_save_path)
    plot_save_path = r'./../../db/plots/predictions/'
    predictions_plot(y_test, predictions, plot_save_path)


# %%
