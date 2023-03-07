import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os


results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="groupadditivity"  # indicate the location of the groupadditivity data directory, available as a zenodo repository. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")
cp = "inferno"
m = ['o', 'x','v','^','s','D','p']


fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=(7,3.5),sharey=True)
#axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_yscale('log')

systems=['noisy/noisy','clean/noisy','clean/clean']
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig1a.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['clean_noisy']==s]
    label = s
    if label == 'clean/noisy':
        label = 'noisy/clean'
    axs[0].scatter(sub['size'],sub['rmse'],marker=m[i],s=20,color=pal[i],label=label)
    axs[0].plot(sub['size'],sub['rmse'],color=pal[i])
axs[0].legend(frameon=False,handletextpad=0.1)
axs[0].set_xlabel("# Datapoints")
axs[0].set_ylabel("RMSE [kcal/mol]")
axs[0].set_title("GDB-11 Add. Enthalpy")

systems=[71,711,7116,71161,711613]
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig1b.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['size']==s]
    axs[1].scatter(sub['noise'],sub['rmse'],marker=m[i],s=20,color=pal[i],label='N='+str(s))
    axs[1].plot(sub['noise'],sub['rmse'],color=pal[i])
    
axs[1].plot(sub['noise'][1:],sub['noise'][1:],"--",color='black')
axs[1].legend(frameon=False,handletextpad=0.1)
axs[1].set_ylim([0.008,30])
axs[1].set_xlabel("Magnitude of noise [kcal/mol]")
#axs[1].set_ylabel("MAE [kcal/mol]")
axs[1].set_title("GDB-11 Add. Enthalpy")

save_path = os.path.join(figures_dir,"figure_1.png")
plt.savefig(save_path,dpi=200)