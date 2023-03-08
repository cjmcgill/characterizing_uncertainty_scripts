import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="data"  # indicate the location of the splits directory, provided with this repo. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")
cp = "inferno"
m = ['o', 'x','v','^','s','D','p']

colors=sns.color_palette(cp,2)

N=[100,300,1000,3000,10000,30000,100000]

true_path=os.path.join(dataset_dir,"u_random_test.csv")
true=pd.read_csv(true_path) #Update to new splits
preds_path=os.path.join(figures_dir,"u_random_100000_preds.csv")
preds=pd.read_csv(preds_path) #Update to 100000
train_path=os.path.join(dataset_dir,"u_random_train_100000.csv")
train=pd.read_csv(train_path) #Update to 100000

true_intrain=true[true['smiles'].isin(train['smiles'].values)]
true_notintrain=true[~true['smiles'].isin(train['smiles'].values)]
pred_intrain=preds[true['smiles'].isin(train['smiles'].values)]
pred_notintrain=preds[~true['smiles'].isin(train['smiles'].values)]

mae1=np.round(mean_absolute_error(true_notintrain['u_atom'],pred_notintrain['u_atom']),2)
mae2=np.round(mean_absolute_error(true_intrain['u_atom'],pred_intrain['u_atom']),2)

intrain_idx=list(true_intrain.index)
np.random.shuffle(intrain_idx)

notintrain_idx=list(true_notintrain.index)
np.random.shuffle(notintrain_idx)

ps=[]
maes=[]
n1=len(intrain_idx)
for n2 in [0,25,50,100,500,750,1000,2000,3000,4000,5000,len(notintrain_idx)]:
    intrain_dev=np.abs(true_intrain.loc[intrain_idx[:n1]]['u_atom']-pred_intrain.loc[intrain_idx[:n1]]['u_atom'])
    notintrain_dev=np.abs(true_notintrain.loc[notintrain_idx[:n2]]['u_atom']-pred_notintrain.loc[notintrain_idx[:n2]]['u_atom'])
    ps.append(n1/(n1+n2)*100)
    maes.append((sum(intrain_dev)+sum(notintrain_dev))/(n1+n2))   
n1=0
n2=len(notintrain_idx)
intrain_dev=np.abs(true_intrain.loc[intrain_idx[:n1]]['u_atom']-pred_intrain.loc[intrain_idx[:n1]]['u_atom'])
notintrain_dev=np.abs(true_notintrain.loc[notintrain_idx[:n2]]['u_atom']-pred_notintrain.loc[notintrain_idx[:n2]]['u_atom'])
ps.append(n1/(n1+n2)*100)
maes.append((sum(intrain_dev)+sum(notintrain_dev))/(n1+n2))


fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=(7,3.5))
axs[0].scatter(ps,[maes[-1]]*len(maes),marker=m[0],s=20,label='True MAE',color=colors[0])
axs[0].scatter(ps,maes,marker=m[1],s=12,label='Perceived MAE',color=colors[1])
axs[0].plot(ps,[maes[-1]]*len(maes),color=colors[0])
axs[0].plot(ps,maes,color=colors[1])
axs[0].set_xlabel("% Test datapoints in train")
axs[0].set_title("QM9 U$_{atom}$(T) 100,000 Datapoints")


n, x, _ = axs[1].hist(np.abs(true_notintrain['u_atom']-pred_notintrain['u_atom']),label='not in train, MAE='+str(mae1),color=colors[0],alpha=0.53,density=True,bins=np.arange(0,6,0.25))
plt.plot(0.5*(x[1:]+x[:-1]),n,color=colors[0])
n, x, _ = axs[1].hist(np.abs(true_intrain['u_atom']-pred_intrain['u_atom']),label='in train, MAE='+str(mae2),color=colors[1],alpha=0.35,density=True,bins=np.arange(0,6,0.25))
plt.plot(0.5*(x[1:]+x[:-1]),n,color=colors[1])

axs[1].set_xlabel("AE [kcal/mol]")
axs[1].set_ylabel("Probability")
axs[1].set_title("QM9 U$_{atom}$(T) 100,000 Datapoints")


axs[0].set_ylabel("MAE [kcal/mol]")
axs[0].legend(frameon=False,handletextpad=0.1)
axs[1].legend(frameon=False,handletextpad=0.1)

save_path=os.path.join(figures_dir,"figure_11.png")
plt.savefig(save_path,dpi=200)