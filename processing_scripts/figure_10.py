#!/usr/bin/python3

import csv
import numpy as np
import os
from rdkit import Chem

results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="data"  # indicate the location of the splits data directory, provided with this github repo. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")


data_sizes = []
data_maes = []
data_rmses = []
data_aggregations = []
data_molecules = []

small_targets = []
small_targets_path = os.path.join(dataset_dir,"test_agg_size_small.csv")
with open(small_targets_path) as f:
    reader = csv.reader(f)
    next(f)
    for line in reader:
        small_targets.append(line[1:])
small_targets = np.array(small_targets, dtype=float)

large_targets = []
large_targets_path = os.path.join(dataset_dir,"test_agg_size_large.csv")
with open(large_targets_path) as f:
    reader = csv.reader(f)
    next(f)
    for line in reader:
        large_targets.append(line[1:])
large_targets = np.array(large_targets, dtype=float)


def chemprop_stats(preds_paths,targets,prop_label):
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


def chemprop_ensemble(preds_paths):
    for i,path in enumerate(preds_paths):
        preds = []
        smiles = []
        with open(path) as f:
            reader=csv.reader(f)
            next(reader)
            for line in reader:
                smiles.append(line[0])
                preds.append(line[1])
        preds = np.array(preds, dtype=float)
        if i == 0:
            sum_preds = preds
        else:
            sum_preds = sum_preds + preds
    preds = sum_preds / len(preds_paths)
    return smiles, preds

# Figure 10ab

for data_size in [100,300,1000,3000,10000,30000,100000]:
    for agg in ["norm","mean"]:
        for molecule in ["large","small"]:
            targets = {"large":large_targets,"small":small_targets}[molecule]

            preds_paths = []
            for seed in [0,1,2,3,4]:
                preds_path = os.path.join(
                    results_dir,
                    f"save_agg_{agg}_size_{data_size}_{seed}",
                    f"preds_{molecule}.csv",
                )
                preds_paths.append(preds_path)
            mae, rmse = chemprop_stats(preds_paths,targets,"enthalpy_H")
            data_sizes.append(data_size)
            data_maes.append(mae)
            data_rmses.append(rmse)
            data_aggregations.append(agg)
            data_molecules.append(molecule)

write_path = os.path.join(figures_dir,"fig10ab.csv")
with open(write_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["molecule","aggregation","data_size","mae","rmse"])
    for i in range(len(data_sizes)):
        writer.writerow([data_molecules[i],data_aggregations[i],data_sizes[i],data_maes[i],data_rmses[i]])

# Figure 10cd

data_aggregations = []
data_ae = []
data_atoms = []

data_size = 100000
for agg in ["norm","mean"]:
    for molecule in ["large","small"]:
        targets = {"large":large_targets,"small":small_targets}[molecule]
        preds_paths = []
        for seed in [0,1,2,3,4]:
            preds_path = os.path.join(
                results_dir,
                f"save_agg_{agg}_size_{data_size}_{seed}",
                f"preds_{molecule}.csv",
            )
            preds_paths.append(preds_path)
        smiles, preds = chemprop_ensemble(preds_paths)
        for i in range(len(smiles)):
            atoms = Chem.MolFromSmiles(smiles[i]).GetNumAtoms()
            data_atoms.append(atoms)
            ae = np.abs(preds[i]-targets[i,2])
            data_ae.append(ae)
            data_aggregations.append(agg)

write_path = os.path.join(figures_dir,"fig10cd.csv")
with open(write_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["aggregation","atoms","ae"])
    for i in range(len(data_ae)):
        writer.writerow([data_aggregations[i],data_ae[i],data_atoms[i]])
