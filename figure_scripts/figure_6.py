import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="groupadditivity"  # indicate the location of the groupadditivity data directory, available as a zenodo repository. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")
cp = "inferno"
m = ['o', 'x','v','^','s','D','p']

data_path = os.path.join(figures_dir,"fig6.csv")
data = pd.read_csv(data_path)

N=[100,300,1000,3000,10000,30000,100000]
results={
        'gap':{
            'fp10':data[(data["property"]=="gap")&(data["model"]=="fp10")].sort_values("data_size")["mae"],
            'fp100':data[(data["property"]=="gap")&(data["model"]=="fp100")].sort_values("data_size")["mae"],
            'fp1000':data[(data["property"]=="gap")&(data["model"]=="fp1000")].sort_values("data_size")["mae"],
            'mpnn C,N,O same':data[(data["property"]=="gap")&(data["model"]=="mpnn_CNO")].sort_values("data_size")["mae"],
            'mpnn C,N same':data[(data["property"]=="gap")&(data["model"]=="mpnn_CN")].sort_values("data_size")["mae"],
            'mpnn':data[(data["property"]=="gap")&(data["model"]=="mpnn")].sort_values("data_size")["mae"],
        },
        'enthalpy_H':{
            'fp10':data[(data["property"]=='enthalpy_H')&(data["model"]=="fp10")].sort_values("data_size")["mae"],
            'fp100':data[(data["property"]=='enthalpy_H')&(data["model"]=="fp100")].sort_values("data_size")["mae"],
            'fp1000':data[(data["property"]=='enthalpy_H')&(data["model"]=="fp1000")].sort_values("data_size")["mae"],
            'mpnn C,N,O same':data[(data["property"]=='enthalpy_H')&(data["model"]=="mpnn_CNO")].sort_values("data_size")["mae"],
            'mpnn C,N same':data[(data["property"]=='enthalpy_H')&(data["model"]=="mpnn_CN")].sort_values("data_size")["mae"],
            'mpnn':data[(data["property"]=='enthalpy_H')&(data["model"]=="mpnn")].sort_values("data_size")["mae"],
        },
    }

labels=['FFN FP 10','FFN FP 100','FFN FP 1000','d-MPNN C,N,O','d-MPNN C,N','d-MPNN',
        'd-MPNN ring'
       ]
colors=sns.color_palette(cp,len(labels))
fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=(7,3.5))
for i, key in enumerate(results['gap'].keys()):
    axs[1].scatter(N,results['gap'][key],marker=m[i],s=20,label=labels[i],color=colors[i])
    axs[1].plot(N,results['gap'][key],color=colors[i])
    
for i, key in enumerate(results['enthalpy_H'].keys()):    
    axs[0].scatter(N,results['enthalpy_H'][key],marker=m[i],s=20,label=labels[i],color=colors[i])
    axs[0].plot(N,results['enthalpy_H'][key],color=colors[i])


axs[1].set_xlabel("# Datapoints")
#axs[1].set_ylabel("MAE [eV]")
axs[1].set_ylim([0.08,1])
axs[0].set_xlabel("# Datapoints")
axs[0].set_ylabel("MAE [eV]")
axs[0].legend(frameon=False,handletextpad=0.1)
#axs[1].legend()
axs[1].set_title("QM9 Gap")
axs[0].set_title("QM9 Enthalpy H")
axs[0].set_ylim([0.1,1000])

axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_yscale('log')

save_path=os.path.join(figures_dir,"figure_6.png")
plt.savefig(save_path,dpi=200)