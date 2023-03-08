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

data_path = os.path.join(figures_dir,"fig5.csv")
data = pd.read_csv(data_path)

#Schnet vs Chemprop 
colors=sns.color_palette(cp,4)
N=[100,300,1000,3000,10000,30000,100000]
results={
    'schnet':{
        'gap':{
            'avg':data[(data["model"]=="schnet") & (data["property"]=="gap") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
            'sum':data[(data["model"]=="schnet") & (data["property"]=="gap") & (data["aggregation"]=="sum")].sort_values("data_size")["mae"]
        },
        'enthalpy_H':{
            'avg':data[(data["model"]=="schnet") & (data["property"]=="enthalpy_H") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
            'sum':data[(data["model"]=="schnet") & (data["property"]=="enthalpy_H") & (data["aggregation"]=="sum")].sort_values("data_size")["mae"]
        }},
    'chemprop':{
        'gap':{
            'avg':data[(data["model"]=="chemprop") & (data["property"]=="gap") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
            'sum':data[(data["model"]=="chemprop") & (data["property"]=="gap") & (data["aggregation"]=="sum")].sort_values("data_size")["mae"]
        },
        'enthalpy_H':{
            'avg':data[(data["model"]=="chemprop") & (data["property"]=="enthalpy_H") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
            'sum':data[(data["model"]=="chemprop") & (data["property"]=="enthalpy_H") & (data["aggregation"]=="sum")].sort_values("data_size")["mae"]
        }},
    }

fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=(7,3.5))
axs[0].scatter(N,results['chemprop']['gap']['avg'],marker=m[0],s=20,label="d-MPNN avg",color=colors[0])
axs[0].scatter(N,results['chemprop']['gap']['sum'],marker=m[1],s=20,label="d-MPNN sum",color=colors[1])
axs[0].scatter(N,results['schnet']['gap']['avg'],marker=m[2],s=20,label="SchNet avg",color=colors[2])
axs[0].scatter(N,results['schnet']['gap']['sum'],marker=m[3],s=20,label="SchNet sum",color=colors[3])

axs[1].scatter(N,results['chemprop']['enthalpy_H']['avg'],marker=m[0],s=20,label="d-MPNN avg",color=colors[0])
axs[1].scatter(N,results['chemprop']['enthalpy_H']['sum'],marker=m[1],s=20,label="d-MPNN sum",color=colors[1])
axs[1].scatter(N,results['schnet']['enthalpy_H']['avg'],marker=m[2],s=20,label="SchNet avg",color=colors[2])
axs[1].scatter(N,results['schnet']['enthalpy_H']['sum'],marker=m[3],s=20,label="SchNet sum",color=colors[3])



axs[0].plot(N,results['chemprop']['gap']['avg'],'--',color=colors[0])
axs[0].plot(N,results['chemprop']['gap']['sum'],color=colors[1])
axs[0].plot(N,results['schnet']['gap']['avg'],'--',color=colors[2])
axs[0].plot(N,results['schnet']['gap']['sum'],color=colors[3])

axs[1].plot(N,results['chemprop']['enthalpy_H']['avg'],'--',color=colors[0])
axs[1].plot(N,results['chemprop']['enthalpy_H']['sum'],color=colors[1])
axs[1].plot(N,results['schnet']['enthalpy_H']['avg'],'--',color=colors[2])
axs[1].plot(N,results['schnet']['enthalpy_H']['sum'],color=colors[3])



axs[0].set_ylim([0.05,1])
axs[0].set_xlabel("# Datapoints")
#axs[1].set_ylabel("MAE [eV]")

axs[1].set_xlabel("# Datapoints")
axs[0].set_ylabel("MAE [eV]")
axs[0].legend(frameon=False,handletextpad=0.1)#loc='lower left')
#axs[0].legend()
axs[0].set_title("QM9 HOMO-LUMO Gap")
axs[1].set_title("QM9 Enthalpy H")

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_yscale('log')

save_path = os.path.join(figures_dir,"figure_5.png")
plt.savefig(save_path,dpi=200)