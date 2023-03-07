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


class cosh_dis(rv_continuous):
    def __init__(self,loc):
        super().__init__(a=-loc,b=loc)
        self.loc = loc
        self.scale = 2*np.sinh(self.loc)
    def _pdf(self,x):
        return np.cosh(x)/self.scale
    
#Noise distributions:
noise_gaussian = [np.random.normal(scale=1) for i in range(100000)]

distribution = cosh_dis(1.543404638418213)
noise_cosh=[distribution.rvs() for i in range(100000)]

distribution = stats.uniform(-np.sqrt(3),2*np.sqrt(3))
noise_uni=[distribution.rvs() for i in range(100000)]

noise_bimodal=[np.random.normal(scale=0.4359)+random.choice([-0.9,0.9]) for i in range(100000)]

fig = plt.figure(figsize=(8,3.25))

systems=['gauss', 'cosh', 'uni', 'bimod', 'clean']
pal=sns.color_palette(cp,len(systems))

grid = gridspec.GridSpec(2, 20, wspace=0.6, hspace=0.7, figure=fig)
axs=(plt.subplot(grid[0, :4],), plt.subplot(grid[0, 4:8]), plt.subplot(grid[1,:4]), plt.subplot(grid[1, 4:8]), plt.subplot(grid[0:, 11:]))

#Gaussian
axs[0].hist(noise_gaussian,bins=100,color=pal[0])
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_title("Gaussian")

#Cosine
axs[1].hist(noise_cosh,bins=100,color=pal[1])
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_title("Cosh")

#Uni
axs[2].hist(noise_uni,bins=100,color=pal[2])
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_title("Uniform")

#Bimodal
axs[3].hist(noise_bimodal,bins=100,color=pal[3])
axs[3].set_xticks([])
axs[3].set_yticks([])
axs[3].set_title("Bimodal")


axs[4].set_yscale('log')
axs[4].set_xscale('log')

data_path = os.path.join(figures_dir,"fig2b.csv")
data=pd.read_csv(data_path)
for i,s in enumerate(systems):
    sub=data[data['noise_type']==s]
    axs[4].scatter(sub['size'],sub['rmse'],marker=m[i],s=20,color=pal[i],label=str(s))
    axs[4].plot(sub['size'],sub['rmse'],color=pal[i])
axs[4].legend(frameon=False,handletextpad=0.1)
axs[4].set_xlabel("# Datapoints")
axs[4].set_ylabel("RMSE [kcal/mol]")
axs[4].set_title("GDB-11 Add. Enthalpy")

save_path = os.path.join(figures_dir,"figure_2.png")
plt.savefig(save_path,dpi=200)
