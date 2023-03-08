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


fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=(7,3.5),sharey=True,sharex=True)
labels=['reported, Eq. (7)','independent test set, Eq. (8)']
systems=['reported','true']
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig9a.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['reported_or_true']==s]
    axs[0].scatter(sub['noise'],sub['rmse'],marker=m[i],s=20,color=pal[i],label=labels[i])
    axs[0].plot(sub['noise'],sub['rmse'],color=pal[i])
axs[0].legend(frameon=False,handletextpad=0.1)
axs[0].set_xlabel("# Folds")
axs[0].set_ylabel("RMSE [kcal/mol]")
axs[0].set_title("GDB-11 Add. Enthalpy")
axs[0].set_ylim([0.07,0.105])
axs[0].set_xlim([0,30])

systems=['reported','true']
labels=['reported, Eq. (2)','independent test set, Eq. (2)']
pal=sns.color_palette(cp,len(systems))
data_path = os.path.join(figures_dir,"fig9b.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['reported_or_true']==s]
    axs[1].scatter(sub['noise'],sub['rmse'],marker=m[i],s=20,color=pal[i],label=labels[i])
    axs[1].plot(sub['noise'],sub['rmse'],color=pal[i])
axs[1].legend(frameon=False,handletextpad=0.1)
axs[1].set_xlabel("# Ensemble Models")
#axs[0].set_ylabel("RMSE [kcal/mol]")
axs[1].set_title("GDB-11 Add. Enthalpy")

save_path = os.path.join(figures_dir,"figure_9.png")
plt.savefig(save_path,dpi=200)