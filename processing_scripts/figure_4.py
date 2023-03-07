#!/usr/bin/python3

# This script collects the error and nonvariance error data needed for Figure 4.

import numpy as np
import csv
import os

results_dir = "results"  # indicate the location of the directory where your trained models and results are stuored
dataset_dir = "groupadditivity_h298"  # indicate the location of the group additivity dataset directory
figures_dir = os.path.join(results_dir, "figures")

fractions = [0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1,1e-05,2e-05,4e-05,6e-05,8e-05,0.0001,0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008]
sizes = [20,50,100,1000,1500,2000]
seeds = [1,2,3,4,5]

header = [
    'directory',
    'size',
    'fraction',
    'data_size',
    'ens_mae',
    'ens_rmse',
    '1_mae',
    '2_mae',
    '3_mae',
    '4_mae',
    '5_mae',
    'nonvariance_mae',
]


def main():
    write_data=[]
    for fraction in fractions:
        for size in sizes:

            preds_dir = os.path.join(results_dir, f"size{size}", str(fraction))
            if not os.path.exists(os.path.join(preds_dir,"test_preds.csv")):
                continue

            line=[preds_dir,size,fraction, int(float(fraction)*7116130)]
            target_path = os.path.join(dataset_dir, "dataset", "groupadditivity_test.csv")
            line.extend(get_stats(preds_dir,target_path))
            write_data.append(line)

    with open(os.path.join(figures_dir, "fig4.csv"),'w') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(write_data)


def get_stats(dir_path,target_path):
    targets = get_targets(target_path)
    preds_path = os.path.join(dir_path,'test_preds.csv')
    if os.path.exists(preds_path):
        preds = get_preds(preds_path)
        ens_mae = np.mean(np.abs(targets-preds))
        ens_rmse = np.sqrt(np.mean(np.square(targets-preds)))
        nonvariance_mae = get_nonvariance(os.path.join(dir_path,'projection.csv'))
    else:
        ens_mae = ''
        ens_rmse = ''
        nonvariance_mae = ''
    model_maes = []
    for seed in seeds:
        model_path = os.path.join(dir_path,str(seed),'test_preds.csv')
        if os.path.exists(model_path):
            model_preds,_ = get_preds(model_path,False)
            model_mae = np.mean(np.abs(targets-model_preds))
            model_maes.append(model_mae)
        else:
            model_maes.append('')
    stats=[ens_mae,ens_rmse,*model_maes,nonvariance_mae]
    return stats


def get_preds(path):
    preds=[]
    with open(path) as f:
        reader=csv.reader(f)
        next(reader)
        for line in reader:
            preds.append(float(line[1]))
    return np.array(preds)


def get_targets(path):
    targets=[]
    with open(path) as f:
        reader=csv.reader(f)
        next(reader)
        for line in reader:
            targets.append(float(line[1]))
    return np.array(targets)


def get_nonvariance(path):
    if not os.path.exists(path):
        return ''
    with open(path) as f:
        reader=csv.reader(f)
        next(reader)
        return float(next(reader)[3])


if __name__ == '__main__':
    main()