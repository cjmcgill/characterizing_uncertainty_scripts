#!/usr/bin/python3

import csv
import numpy as np
import os

results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="data"  # indicate the location of the splits data directory, provided with this github repo. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")


data_sizes = []
data_maes = []
data_rmses = []
data_models = []
data_prop = []

targets = []
targets_path = os.path.join(dataset_dir,"test.csv")
with open(targets_path) as f:
    reader = csv.reader(f)
    next(f)
    for line in reader:
        targets.append(line[1:])
targets = np.array(targets, dtype=float)


def chemprop_stats(preds_paths,prop_label):
    idx = {"enthalpy_H":2, "gap":1}[prop_label]
    for i,path in enumerate(preds_paths):
        preds = []
        with open(path) as f:
            reader=csv.reader(f)
            next(reader)
            for line in reader:
                preds.append(line[1])
        preds = np.array(preds, dtype=float)
        if i == 0:
            sum_preds = preds
        else:
            sum_preds = sum_preds + preds
    preds = sum_preds / len(preds_paths)
    error = preds - targets[:,idx]
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))
    return mae, rmse


def schnet_stats(paths,prop_label):
    idx = {"enthalpy_H":2, "gap":1}[prop_label]
    for i,path in enumerate(paths):
        preds = []
        with open(path) as f:
            reader=csv.reader(f)
            next(reader)
            for line in reader:
                preds.append(line[0])
        preds = np.array(preds, dtype=float)
        if i == 0:
            sum_preds = preds
        else:
            sum_preds = sum_preds + preds
    preds = sum_preds / len(paths)
    error = preds - targets[:,idx]
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))
    return mae, rmse

# Figure 6

for data_size in [100,300,1000,3000,10000,30000,100000]:

    for prop in ["gap","enthalpy_H"]:

        # mpnn
        preds_paths = []
        for seed in [0,1,2,3,4]:
            preds_path = os.path.join(
                results_dir,
                f"save_{prop}_norm_{data_size}_{seed}",
                "test_preds.csv",
            )
            preds_paths.append(preds_path)
        mae, rmse = chemprop_stats(preds_paths,prop)
        data_sizes.append(data_size)
        data_maes.append(mae)
        data_rmses.append(rmse)
        data_models.append("mpnn")
        data_prop.append(prop)

        # mpnn with custom descriptors
        for features in ["CN","CNO"]:
            preds_paths = []
            for seed in [0,1,2,3,4]:
                preds_path = os.path.join(
                    results_dir,
                    f"save_{prop}_norm_{features}_{data_size}_{seed}",
                    "test_preds.csv",
                )
                preds_paths.append(preds_path)
            mae, rmse = chemprop_stats(preds_paths,prop)
            data_sizes.append(data_size)
            data_maes.append(mae)
            data_rmses.append(rmse)
            data_models.append(f"mpnn_{features}")
            data_prop.append(prop)

        # fp
        for fp in [10,100,1000]:
            preds_paths = []
            for seed in [0,1,2,3,4]:
                preds_path = os.path.join(
                    results_dir,
                    f"save_{prop}_fp{fp}_{data_size}_{seed}",
                    "test_preds.csv",
                )
                preds_paths.append(preds_path)
            mae, rmse = chemprop_stats(preds_paths,prop)
            data_sizes.append(data_size)
            data_maes.append(mae)
            data_rmses.append(rmse)
            data_models.append(f"fp{fp}")
            data_prop.append(prop)

write_path = os.path.join(figures_dir,"fig6.csv")
with open(write_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["property","model","data_size","mae","rmse"])
    for i in range(len(data_sizes)):
        writer.writerow([data_prop[i],data_models[i],data_sizes[i],data_maes[i],data_rmses[i]])




