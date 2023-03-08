import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from scipy.stats import rv_continuous
from scipy import stats
import random


results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="groupadditivity"  # indicate the location of the groupadditivity data directory, available as a zenodo repository. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")
cp = "inferno"
m = ['o', 'x','v','^','s','D','p']



fig, axs = plt.subplots(2, 2, tight_layout=True,figsize=(7,6.5),sharex=True,sharey=True)


axs[0][0].set_yscale('log')
axs[0][1].set_yscale('log')
axs[1][0].set_yscale('log')
axs[1][1].set_yscale('log')

systems=[711,7116,711613]
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig7a.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['size']==s]
    axs[0][0].scatter(sub['noise'],sub['mae'],marker=m[i],s=20,color=pal[i],label='N='+str(s))
    axs[0][0].plot(sub['noise'],sub['mae'],color=pal[i])
axs[0][0].legend(frameon=False,handletextpad=0.1, loc=[0.6,0.65])
#axs[0][0].set_xlabel("# Ensembles")
axs[0][0].set_ylabel("MAE [kcal/mol]")
axs[0][0].set_title("GDB-11 Add. Enthalpy")

systems=[20,100,1000]
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig7b.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['hidden_size']==s]
    axs[0][1].scatter(sub['noise'],sub['mae'],marker=m[i],s=20,color=pal[i],label='h='+str(s))
    axs[0][1].plot(sub['noise'],sub['mae'],color=pal[i])
axs[0][1].legend(frameon=False,handletextpad=0.1)
#axs[0][1].set_xlabel("# Ensembles")
#axs[0][1].set_ylabel("MAE [kcal/mol]")
axs[0][1].set_title("GDB-11 Add. Enthalpy")

systems=[1,0.02,0]
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig7c.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['ensemble_size']==s]
    axs[1][0].scatter(sub['noise'],sub['mae'],marker=m[i],s=20,color=pal[i],label='noise='+str(s))
    axs[1][0].plot(sub['noise'],sub['mae'],color=pal[i])
axs[1][0].legend(frameon=False,handletextpad=0.1)
axs[1][0].set_xlabel("# Ensemble Models")
axs[1][0].set_ylabel("MAE [kcal/mol]")
#axs[1][0].set_title("GDB-11 Add. Enthalpy")

systems=['mean','norm']
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig7d.csv")
data=pd.read_csv(data_path)
offset={'mean':0,'norm':1}
for i,s in enumerate(systems):
    sub=data[data['aggregation']==s]
    axs[1][1].scatter(sub['noise']+offset[s],sub['mae'],marker=m[i],s=20,color=pal[i],label='aggregation '+s)
    axs[1][1].plot(sub['noise']+offset[s],sub['mae'],color=pal[i])
axs[1][1].legend(frameon=False,handletextpad=0.1)
axs[1][1].set_xlabel("# Ensemble Models")
#axs[1][1].set_ylabel("MAE [kcal/mol]")
#axs[1][1].set_title("GDB-11 Add. Enthalpy")

save_path = os.path.join(figures_dir,"figure_7.png")
plt.savefig(save_path,dpi=200)