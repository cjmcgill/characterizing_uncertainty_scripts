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

data_path = os.path.join(figures_dir, "fig4.csv")
data = pd.read_csv(data_path)

fig, axs = plt.subplots(2, 2, tight_layout=True,figsize=(7,6),sharex=True)

axs[0][0].set_xscale('log')
axs[0][0].set_yscale('log')
axs[1][0].set_yscale('log')
axs[1][1].set_yscale('log')
axs[0][0].set_ylim([0.005,50])
axs[1][0].set_ylim([0.005,50])
axs[1][1].set_ylim([0.005,50])

systems=[20, 50, 100, 200, 1000, 1500, 2000]
pal=sns.color_palette(cp,len(systems))
for i,s in enumerate(systems):
    sub=data[data['size']==s].sort_values('data_size')
    
    axs[0][0].scatter(sub['data_size'],sub['ens_mae'],marker=m[i],s=20,color=pal[i],label="h="+str(s))
    axs[0][0].plot(sub['data_size'],sub['ens_mae'],color=pal[i])

    sub = sub[sub['fraction_nonvariance'].notna()] #For gaps, remove later
    
    axs[0][1].scatter(sub['data_size'],sub['fraction_nonvariance'],marker=m[i],s=12,color=pal[i],label="h="+str(s))
    axs[0][1].plot(sub['data_size'],sub['fraction_nonvariance'],color=pal[i])
    
s=2000
sub=data[data['size']==s].sort_values('data_size')
axs[1][0].scatter(sub['data_size'],sub['ens_mae'],marker=m[-1],s=20,color=pal[-1])
axs[1][0].plot(sub['data_size'],sub['ens_mae'],color=pal[-1])
sub = sub[sub['fraction_nonvariance'].notna()] #For gaps, remove later
axs[1][0].plot(sub['data_size'],sub['fraction_nonvariance']*sub['ens_mae'],":",color=pal[-1])
axs[1][0].plot(sub['data_size'],(1-sub['fraction_nonvariance'])*sub['ens_mae'],'--',color=pal[-1])

s=20
sub=data[data['size']==s].sort_values('data_size')
axs[1][1].scatter(sub['data_size'],sub['ens_mae'],marker=m[0],s=20,color=pal[0])
axs[1][1].plot(sub['data_size'],sub['ens_mae'],color=pal[0])
sub = sub[sub['fraction_nonvariance'].notna()] #For gaps, remove later
axs[1][1].plot(sub['data_size'],sub['fraction_nonvariance']*sub['ens_mae'],":",color=pal[0])
axs[1][1].plot(sub['data_size'],(1-sub['fraction_nonvariance'])*sub['ens_mae'],'--',color=pal[0])

axs[1][1].scatter([np.nan,np.nan],[np.nan,np.nan],marker=m[i],s=20,color='black', label='total')        
axs[1][1].plot([np.nan,np.nan],[np.nan,np.nan],":",color='black', label='bias')
axs[1][1].plot([np.nan,np.nan],[np.nan,np.nan],"--",color='black', label='variance')

axs[1][1].set_xlabel("# Datapoints")
axs[0][0].set_ylabel("MAE [kcal/mol]")
axs[1][0].set_ylabel("MAE [kcal/mol]")
axs[1][1].set_ylabel("MAE [kcal/mol]")
axs[0][0].set_title("GDB-11 Add. Enthalpy")

axs[0][1].legend(frameon=False,handletextpad=0.1)
axs[1][1].legend(frameon=False,handletextpad=0.1,loc='lower left')
axs[1][0].set_xlabel("# Datapoints")
axs[0][1].set_ylabel("Fraction of error NOT from variance")
axs[0][1].set_title("GDB-11 Add. Enthalpy")

axs[1][0].text(100000,20,"h=2000")
axs[1][1].text(200000,20,"h=20")
save_path = os.path.join(figures_dir,"figure_4.png")
plt.savefig(save_path,dpi=200)