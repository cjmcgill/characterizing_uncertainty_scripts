import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="groupadditivity"  # indicate the location of the groupadditivity data directory, available as a zenodo repository. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")


def compute_auce(preds_dir, true, index, n_ensemble, pred=[], unce=[], q=10):
    if len(pred) ==0 or len(unce)==0:
        for i in range(n_ensemble):
            preds_path = os.path.join(preds_dir,f"preds_{i}.csv")
            new = pd.read_csv(preds_path)['h298'].values[index].reshape(-1,1)
            if i ==1:
                current = new
            else:
                current = np.concatenate((current,new),axis=1)
        pred = np.mean(current,axis=1)
        unce = np.std(current,axis=1)**2
    
    calibration_x = []
    calibration_y = []
    calibration_y_oracle = []
    auce = 0

    f = n_ensemble-1
    for i in range(q):
        upper = (i+1)*1/q
        ctr = 0
        for u,p,t in zip(unce,pred,true):
            s = np.sqrt(u)/np.sqrt(f)
            i = stats.t.interval(alpha=upper, df=f, loc=p, scale=s) 
            if t>= i[0] and t <= i[1]:
                ctr+=1
        calibration_x.append(upper)
        calibration_y.append(ctr/len(unce))
        calibration_y_oracle.append(upper)
        auce += abs(ctr/len(unce) - upper) * 1/q
    return auce, calibration_x, calibration_y, calibration_y_oracle

n=10000
n_ensemble = 5

target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")
df = pd.read_csv(target_path)
index = df.sample(n, random_state=0).index
true = df['h298'].values[index]


fig, axs = plt.subplots(2,4,figsize=(8,5),sharey=True)


system = 'default'
path = os.path.join(results_dir,"ensemble_base")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[0][0].plot(calibration_x,calibration_y,color='darkorange')
axs[0][0].plot(calibration_x,calibration_y_oracle,color='gray')
axs[0][0].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[0][0].set_title(system,fontsize=10)
axs[0][0].text(0.15,0.9,"AUCE "+str(np.round(auce,4)))
print(system, auce)
best=auce

#Systems:
system = 'N = 711'
path = os.path.join(results_dir,"ensemble_fraction0.0001")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[0][1].plot(calibration_x,calibration_y,color='darkorange')
axs[0][1].plot(calibration_x,calibration_y_oracle,color='gray')
axs[0][1].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[0][1].set_title(system,fontsize=10)
axs[0][1].text(0.15,0.9,"AUCE "+str(np.round(auce,2)))
print(system, auce)

system = 'N = 7116'
path = os.path.join(results_dir,"ensemble_fraction0.001")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[0][2].plot(calibration_x,calibration_y,color='darkorange')
axs[0][2].plot(calibration_x,calibration_y_oracle,color='gray')
axs[0][2].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[0][2].set_title(system,fontsize=10)
axs[0][2].text(0.15,0.9,"AUCE "+str(np.round(auce,2)))
print(system, auce)

system = 'h = 20'
path = os.path.join(results_dir,"ensemble_size20")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[0][3].plot(calibration_x,calibration_y,color='darkorange')
axs[0][3].plot(calibration_x,calibration_y_oracle,color='gray')
axs[0][3].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[0][3].set_title(system,fontsize=10)
axs[0][3].text(0.15,0.9,"AUCE "+str(np.round(auce,3)))
print(system, auce)

system = 'h = 100'
path = os.path.join(results_dir,"ensemble_size100")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[1][0].plot(calibration_x,calibration_y,color='darkorange')
axs[1][0].plot(calibration_x,calibration_y_oracle,color='gray')
axs[1][0].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[1][0].set_title(system,fontsize=10)
axs[1][0].text(0.15,0.9,"AUCE "+str(np.round(auce,4)))
print(system, auce)

system = 'noise = 0.02'
path = os.path.join(results_dir,"ensemble_noise0.02")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[1][1].plot(calibration_x,calibration_y,color='darkorange')
axs[1][1].plot(calibration_x,calibration_y_oracle,color='gray')
axs[1][1].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[1][1].set_title(system,fontsize=10)
axs[1][1].text(0.15,0.9,"AUCE "+str(np.round(auce,4)))
print(system, auce)

system = 'noise = 1'
path = os.path.join(results_dir,"ensemble_noise1")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[1][2].plot(calibration_x,calibration_y,color='darkorange')
axs[1][2].plot(calibration_x,calibration_y_oracle,color='gray')
axs[1][2].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[1][2].set_title(system,fontsize=10)
axs[1][2].text(0.15,0.9,"AUCE "+str(np.round(auce,3)))
print(system, auce)

system = 'aggregation mean'
path = os.path.join(results_dir,"ensemble_mean")
auce, calibration_x, calibration_y, calibration_y_oracle = compute_auce(path, true, index, n_ensemble, q=20)
axs[1][3].plot(calibration_x,calibration_y,color='darkorange')
axs[1][3].plot(calibration_x,calibration_y_oracle,color='gray')
axs[1][3].fill_between(calibration_x,calibration_y,calibration_y_oracle, color='lightgray')
axs[1][3].set_title(system,fontsize=10)
axs[1][3].text(0.15,0.9,"AUCE "+str(np.round(auce,3)))
print(system, auce)


axs[1][0].set_xlabel("p")
axs[1][1].set_xlabel("p")
axs[1][2].set_xlabel("p")
axs[1][3].set_xlabel("p")
axs[0][0].set_ylabel("Empirical coverage")
axs[1][0].set_ylabel("Empirical coverage")

plt.tight_layout()
save_path = os.path.join(figures_dir,"figure_8.png")
plt.savefig(save_path, dpi=300)