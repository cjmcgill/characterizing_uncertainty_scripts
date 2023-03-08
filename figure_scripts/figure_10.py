import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="groupadditivity"  # indicate the location of the groupadditivity data directory, available as a zenodo repository. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")
cp = "inferno"
m = ['o', 'x','v','^','s','D','p']

data_path = os.path.join(figures_dir,"fig10ab.csv")
data = pd.read_csv(data_path)

colors=sns.color_palette(cp,2)
N=[100,300,1000,3000,10000,30000,100000]

fig, axs = plt.subplots(2, 2, tight_layout=True,sharey='row',figsize=(7, 6.7))
axs[0][0].scatter(N,
                data[(data["molecule"]=="small") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
                marker=m[0],s=20,label='test #1-6',color=colors[0])
axs[0][0].scatter(N,
                data[(data["molecule"]=="large") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
                marker=m[1],s=20,label='test #7-9',color=colors[1])
axs[0][0].plot(N,
                data[(data["molecule"]=="small") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
                color=colors[0])
axs[0][0].plot(N,
                data[(data["molecule"]=="large") & (data["aggregation"]=="mean")].sort_values("data_size")["mae"],
                color=colors[1])

axs[0][1].scatter(N,
                data[(data["molecule"]=="small") & (data["aggregation"]=="norm")].sort_values("data_size")["mae"],
                marker=m[0],s=20,label='test #1-6',color=colors[0])
axs[0][1].scatter(N,
                data[(data["molecule"]=="large") & (data["aggregation"]=="norm")].sort_values("data_size")["mae"],
                marker=m[1],s=20,label='test #7-9',color=colors[1])
axs[0][1].plot(N,
                data[(data["molecule"]=="small") & (data["aggregation"]=="norm")].sort_values("data_size")["mae"],
                color=colors[0])
axs[0][1].plot(N,
                data[(data["molecule"]=="large") & (data["aggregation"]=="norm")].sort_values("data_size")["mae"],
                color=colors[1])

data_path = os.path.join(figures_dir,"fig10cd.csv")
data = pd.read_csv(data_path)


axs[1][0].scatter(data[(data["aggregation"]=="mean")&(data["atoms"]<7)]["atoms"],data[(data["aggregation"]=="mean")&(data["atoms"]<7)]["ae"],label='test #1-6',color=colors[0],alpha=0.5)
axs[1][0].scatter(data[(data["aggregation"]=="mean")&(data["atoms"]>6)]["atoms"],data[(data["aggregation"]=="mean")&(data["atoms"]>6)]["ae"],label='test #7-9',color=colors[1],alpha=0.5)

axs[1][1].scatter(data[(data["aggregation"]=="norm")&(data["atoms"]<7)]["atoms"],data[(data["aggregation"]=="norm")&(data["atoms"]<7)]["ae"],label='test #1-6',color=colors[0],alpha=0.5)
axs[1][1].scatter(data[(data["aggregation"]=="norm")&(data["atoms"]>6)]["atoms"],data[(data["aggregation"]=="norm")&(data["atoms"]>6)]["ae"],label='test #7-9',color=colors[1],alpha=0.5)


axs[1][1].set_xlabel("# Atoms")
axs[1][0].set_xlabel("# Atoms")
axs[1][0].set_ylabel("Absolute Error [eV]")
axs[1][0].set_title("MAE for N=100000, Avg")

axs[0][0].set_xlabel("# Datapoints")
axs[0][1].set_xlabel("# Datapoints")
axs[0][0].set_ylabel("MAE [eV]")

axs[0][0].set_title("QM9 Enthalpy H, Avg")

axs[0][0].set_xscale('log')
axs[0][0].set_yscale('log')

axs[1][1].set_title("MAE for N=100000, Sum")

axs[0][1].legend(frameon=False,handletextpad=0.1)
axs[1][1].legend(frameon=False,handletextpad=0.1)
axs[0][1].set_title("QM9 Enthalpy H, Sum")

axs[0][1].set_xscale('log')
axs[0][1].set_yscale('log')

save_path = os.path.join(figures_dir,"figure_10.png")
plt.savefig(save_path,dpi=200)