#!/usr/bin/python3

import csv
import numpy as np
import os

results_dir = "results"  # indicate the location of the directory where your trained models and results are stuored
dataset_dir = "groupadditivity_h298"  # indicate the location of the group additivity dataset directory
figures_dir = os.path.join(results_dir, "figures")

fractions=['1e-05','2e-05','4e-05','6e-05','8e-05','0.0001','0.0002','0.0004','0.0006','0.0008','0.001','0.002','0.004','0.006','0.008','0.01','0.02','0.04','0.06','0.08','0.1']
data_sizes=[int(float(i)*7116130) for i in fractions]

fig1b=[
    ['0.01','0.02','0.08','0.2','0.4','1'],
    ['0.1','0.01','0.001','0.0001','1e-05']
]

fig2b=[
    ['size1000',os.path.join(dataset_dir,"dataset","groupadditivity_test.csv"),'clean'],
    ['noise1_uni',os.path.join(dataset_dir,"dataset","noise1_uniform","groupadditivity_test_noise1_uni.csv"),'uni'],
    ['noise1_cosh',os.path.join(dataset_dir,"dataset","noise1_cosh","groupadditivity_test_noise1_cosh.csv"),'cosh'],
    ['noise1_bimod',os.path.join(dataset_dir,"dataset","noise1_bimod","groupadditivity_test_noise1_bimod.csv"),'bimod'],
    ['noise1',os.path.join(dataset_dir,"dataset","noise1","groupadditivity_test_noise1.csv"),'gauss']
]

fig7a=[
    ['0.0001','0.001'],
    'ensemble_base'
]

fig7b=[
    ['20','100'],
    'ensemble_base'
]

fig7c=[
    ['0.02','1'],
    'ensemble_base'
]

fig7d=[
    'ensemble_mean',
    'ensemble_base'
]

def main():
    collect_fig1a()
    collect_fig1b()
    collect_fig2b()
    collect_fig7a()
    collect_fig7b()
    collect_fig7c()
    collect_fig7d()
    collect_fig9a()
    collect_fig9b()


def collect_fig9b():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    preds_paths=[]
    ensemble_sizes=[]
    folds=[]

    preds_paths=[]
    observed_maes = []
    observed_rmses = []
    for i in range(30):

        preds_dir= "folds_comparison/ensemble"
        preds_path= os.path.join(results_dir, preds_dir, f"test_preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        folds.append("ensemble")

    preds_paths=[]
    for i in range(20):

        preds_dir= "folds_comparison/ensemble"
        preds_path= os.path.join(results_dir, preds_dir, f"independent_preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(np.mean(mae))
        rmses.append(np.mean(rmse))
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        folds.append("independent")

    write_path=os.path.join(figures_dir,'fig9b.csv')
    write_data(write_path,dirs,maes,rmses,sizes,None,ensemble_sizes,'fold',folds)


def collect_fig9a():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    preds_paths=[]
    ensemble_sizes=[]
    folds=[]

    preds_paths=[]
    observed_maes = []
    observed_rmses = []
    for i in range(30):

        preds_dir= "folds_comparison/cv"
        preds_path= os.path.join(results_dir, preds_dir, f"test_preds_{i}.csv")
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=get_stats(preds_path,target_path)
        observed_maes.append(mae)
        observed_rmses.append(rmse)
        dirs.append(preds_dir)
        maes.append(np.mean(observed_maes))
        rmses.append(np.mean(observed_rmses))
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        folds.append("crossval")

    preds_paths=[]
    for i in range(20):

        preds_dir= "folds_comparison/cv"
        preds_path= os.path.join(results_dir, preds_dir, f"independent_preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(np.mean(mae))
        rmses.append(np.mean(rmse))
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        folds.append("independent")

    write_path=os.path.join(figures_dir,'fig9a.csv')
    write_data(write_path,dirs,maes,rmses,sizes,None,ensemble_sizes,'fold',folds)


def collect_fig7d():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    aggregation=[]
    preds_paths=[]
    ensemble_sizes=[]

    preds_paths=[]
    for i in range(20):

        preds_dir=fig7d[0]
        preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        aggregation.append("mean")

    preds_paths=[]
    for i in range(20):

        preds_dir=fig7d[1]
        preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        aggregation.append("norm")
    write_path=os.path.join(figures_dir,'fig7d.csv')
    write_data(write_path,dirs,maes,rmses,sizes,None,ensemble_sizes,'aggregation',aggregation)


def collect_fig7c():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    noises=[]
    ensemble_sizes=[]
    for noise in fig7c[0]:
        preds_paths=[]
        for i in range(20):

            preds_dir=os.path.join(f'ensemble_noise{noise}')
            preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
            preds_paths.append(preds_path)
            target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

            mae,rmse=ensemble_stats(preds_paths,target_path)
            dirs.append(preds_dir)
            maes.append(mae)
            rmses.append(rmse)
            sizes.append('0.1')
            ensemble_sizes.append(i+1)
            noises.append(noise)

    preds_paths=[]
    for i in range(20):

        preds_dir = fig7c[1]
        preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        noises.append('0')
    write_path=os.path.join(figures_dir,'fig7c.csv')
    write_data(write_path,dirs,maes,rmses,sizes,noises,ensemble_sizes)


def collect_fig7b():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    hidden_sizes=[]
    ensemble_sizes=[]
    for hidden_size in fig7b[0]:
        preds_paths=[]
        for i in range(20):

            preds_dir=os.path.join(f'ensemble_size{hidden_size}')
            preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
            preds_paths.append(preds_path)
            target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

            mae,rmse=ensemble_stats(preds_paths,target_path)
            dirs.append(preds_dir)
            maes.append(mae)
            rmses.append(rmse)
            sizes.append('0.1')
            ensemble_sizes.append(i+1)
            hidden_sizes.append(hidden_size)

    preds_paths=[]
    for i in range(20):

        preds_dir = fig7b[1]
        preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
        hidden_sizes.append('1000')
    write_path=os.path.join(figures_dir,'fig7b.csv')
    write_data(write_path,dirs,maes,rmses,sizes,None,ensemble_sizes,'hidden_size',hidden_sizes)


def collect_fig7a():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    ensemble_sizes=[]
    for fraction in fig7a[0]:
        preds_paths=[]
        for i in range(20):

            preds_dir=os.path.join(f'ensemble_fraction{fraction}')
            preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
            preds_paths.append(preds_path)
            target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

            mae,rmse=ensemble_stats(preds_paths,target_path)
            dirs.append(preds_dir)
            maes.append(mae)
            rmses.append(rmse)
            sizes.append(fraction)
            ensemble_sizes.append(i+1)

    preds_paths=[]
    for i in range(20):

        preds_dir = fig7a[1]
        preds_path= os.path.join(results_dir, preds_dir, f"preds_{i}.csv")
        preds_paths.append(preds_path)
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=ensemble_stats(preds_paths,target_path)
        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append('0.1')
        ensemble_sizes.append(i+1)
    write_path=os.path.join(figures_dir,'fig7a.csv')
    write_data(write_path,dirs,maes,rmses,sizes,None,ensemble_sizes)


def collect_fig2b():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    noises=[]
    noise_types=[]
    for preds_dir,target_path,noise_type in fig2b:
        for fraction in fractions:
            mae,rmse=get_stats(
                os.path.join(results_dir,preds_dir,fraction,"test_preds.csv"),
                target_path
            )
            dirs.append(preds_dir)
            maes.append(mae)
            rmses.append(rmse)
            sizes.append(fraction)
            if noise_type=='clean':
                noises.append(0)
            else:
                noises.append(1)
            noise_types.append(noise_type)
    write_path=os.path.join(figures_dir,'fig2b.csv')
    write_data(write_path,dirs,maes,rmses,sizes,noises,None,'noise_type',noise_types)


def collect_fig1b():
    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    noises=[]
    for noise in fig1b[0]:
        for fraction in fig1b[1]:
            noise_target_path=os.path.join(dataset_dir,"dataset",f"noise{noise}","groupadditivity_test_noise{noise}.csv")
            preds_dir=os.path.join(results_dir,f"noise{noise}")
            preds_path = os.path.join(preds_dir,str(fraction),"test_preds.csv")
            if not os.path.exists(preds_path):
                continue
            mae,rmse=get_stats(
                preds_path,
                noise_target_path,
            )
            dirs.append(preds_dir)
            maes.append(mae)
            rmses.append(rmse)
            sizes.append(fraction)
            noises.append(noise)
    write_path=os.path.join(figures_dir,'fig1b.csv')
    write_data(write_path,dirs,maes,rmses,sizes,noises)


def collect_fig1a():

    dirs=[]
    maes=[]
    rmses=[]
    sizes=[]
    clean_noise=[]

    for fraction in fractions:

        preds_dir = os.path.join(results_dir,"size1000")
        preds_path = os.path.join(preds_dir, str(fraction), "test_preds.csv")
        if not os.path.exists(preds_path):
            continue
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")

        mae,rmse=get_stats(
            preds_path,
            target_path,
        )

        dirs.append(preds_dir)
        maes.append(mae)
        rmses.append(rmse)
        sizes.append(fraction)
        clean_noise.append('clean/clean')

    for fraction in fractions:

        preds_dir = os.path.join(results_dir,"noise1")
        preds_path = os.path.join(preds_dir,str(fraction),"test_preds.csv")
        if not os.path.exists(preds_path):
            continue
        target_path = os.path.join(dataset_dir,"dataset","groupadditivity_test.csv")
        noise_target_path = os.path.join(dataset_dir,"dataset","noise1","groupadditivity_test_noise1.csv")

        mae,rmse,c_n=noise_stats(
            preds_path,
            target_path,
            noise_target_path,
        )
        dirs.extend([preds_dir]*2)
        maes.extend(mae)
        rmses.extend(rmse)
        sizes.extend([fraction]*2)
        clean_noise.extend(c_n)
    write_path=os.path.join(figures_dir,'fig1a.csv')
    write_data(write_path,dirs,maes,rmses,sizes,None,None,'clean_noisy',clean_noise)


def collect_test_score(scores_path):
    with open(scores_path) as f:
        reader=csv.reader(f)
        next(reader)
        score=float(next(reader)[1])
        return score


def ensemble_stats(preds_paths,target_path):
    targets=load_data(target_path)
    for i,preds_path in enumerate(preds_paths):
        preds=load_data(preds_path)
        if i==0:
            sum_preds=preds
        else:
            sum_preds=sum_preds + preds
    ensemble_preds=sum_preds / len(preds_paths)
    error=ensemble_preds-targets
    mae=np.mean(np.abs(error))
    rmse=np.sqrt(np.mean(np.square(error)))
    return mae,rmse


def get_stats(preds_dir,target_path):
    preds_path=os.path.join(preds_dir,'test_preds.csv')
    targets=load_data(target_path)
    preds=load_data(preds_path)
    error=preds-targets
    return mae_rmse(error)


def large_ensemble_stats(preds_dir,num):
    maes=[]
    rmses=[]
    preds_paths=[os.path.join(preds_dir,'preds_'+str(i)+'.csv') for i in range(num)]
    targets=load_data(target_file)
    for preds_path in preds_paths:
        preds=load_data(preds_path)
        error=targets-preds
        mae,rmse=mae_rmse(error)
        maes.append(mae)
        rmses.append(rmse)
    return maes, rmses


def noise_stats(preds_dir,target_path,noise_test):
    clean_targets=load_data(target_path)
    noisy_targets=load_data(noise_test)
    preds_path=os.path.join(preds_dir,'test_preds.csv')
    preds=load_data(preds_path)
    clean_error=preds-clean_targets
    noisy_error=preds-noisy_targets
    clean_mae,clean_rmse=mae_rmse(clean_error)
    noisy_mae,noisy_rmse=mae_rmse(noisy_error)
    return [clean_mae,noisy_mae],[clean_rmse,noisy_rmse],['clean/noisy','noisy/noisy']


def load_data(path,collect_var=False):
    data=[]
    variances=[]
    with open(path) as f:
        reader=csv.reader(f)
        next(reader)
        for line in reader:
            data.append(float(line[1]))
            if collect_var:
                variances.append(float(line[2]))
    if collect_var:
        return np.array(data),np.array(variances)
    else:
        return np.array(data)


def mae_rmse(error):
    mae=np.mean(np.abs(error))
    rmse=np.sqrt(np.mean(np.square(error)))
    return mae,rmse


def write_data(path,dirs,ensemble_mae,ensemble_rmse,size,noise=None,ensemble_size=None,special_label=None,special=None):
    header=['dir','mae','rmse','size','noise','ensemble_size']
    if special is not None:
        header.append(special_label)
    with open(path,'w') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        for i in range(len(dirs)):
            line=[dirs[i],ensemble_mae[i],ensemble_rmse[i]]
            data_size=data_sizes[fractions.index(size[i])]
            line.append(data_size)
            if noise is None:
                line.append(0)
            else:
                line.append(noise[i])
            if ensemble_size is None:
                line.append(5)
            else:
                line.append(ensemble_size[i])
            if special is not None:
                line.append(special[i])
            writer.writerow(line)
        

if __name__ == '__main__':
    main()