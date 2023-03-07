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

data_path = os.path.join(figures_dir, "fig3.csv")
data = pd.read_csv(data_path)

fig, axs = plt.subplots(2, 2, tight_layout=True,sharey=True, sharex=True, figsize=(7, 6))
m="o"
ms=2.5
cap=2
stds=1
colors=sns.color_palette(cp,2)


axs[0][0].errorbar(
    x=data[data["panel"]=="a"]["target"],
    y=data[data["panel"]=="a"]["pred"],
    yerr=data[data["panel"]=="a"]["bar"],
    fmt=m,
    ms=ms,
    color=colors[0],
    capsize=cap
)
axs[0,0].set_title("Non-Nitrogen-Containing\nLow Noise")
axs[0][1].errorbar(
    x=data[data["panel"]=="b"]["target"],
    y=data[data["panel"]=="b"]["pred"],
    yerr=data[data["panel"]=="b"]["bar"],
    fmt=m,
    ms=ms,
    color=colors[0],
    capsize=cap
)
axs[0,1].set_title("Nitrogen-Containing\nHigh Noise")
axs[1][0].errorbar(
    x=data[data["panel"]=="c"]["target"],
    y=data[data["panel"]=="c"]["pred"],
    yerr=data[data["panel"]=="c"]["bar"],
    fmt=m,
    ms=ms,
    color=colors[1],
    capsize=cap,
)
axs[1,0].set_title("Positive Enthalpy High Noise\nScaled Ensemble Uncertainty")
axs[1][1].errorbar(
    x=data[data["panel"]=="d"]["target"],
    y=data[data["panel"]=="d"]["pred"],
    yerr=data[data["panel"]=="d"]["bar"],
    fmt=m,
    ms=ms,
    color=colors[1],
    capsize=cap,
)
axs[1,1].set_title("Positive Enthalpy High Noise\nMean-Variance Estimation Uncertainty")

axs[0][0].set_xlim([-100,100])
axs[0][0].set_ylim([-100,100])
axs[0][0].set_yticks([-100,-50,0,50,100])
axs[1][0].set_xlabel("Target Enthalpy (kcal/mol)")
axs[1][1].set_xlabel("Target Enthalpy (kcal/mol)")
axs[0][0].set_ylabel("Predicted Enthalpy (kcal/mol)")
axs[1][0].set_ylabel("Predicted Enthalpy (kcal/mol)")

save_path = os.path.join(figures_dir,"figure_3.png")
plt.savefig(save_path, dpi=400)