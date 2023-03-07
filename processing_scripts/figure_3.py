#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
import csv

results_dir = "results"  # indicate the location of the directory where your trained models and results are stuored
dataset_dir = "groupadditivity_h298"  # indicate the location of the group additivity dataset directory
figures_dir = os.path.join(results_dir, "figures")

test_path = os.path.join(dataset_dir, "dataset", "groupadditivity_test.csv")

nitrogen_smiles = []
non_nitrogen_smiles = []
negative_smiles = []
positive_smiles = []

with open(test_path) as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:

        if "n" in line[0] or "N" in line[0]:
            nitrogen_smiles.append(line[0])
        else:
            non_nitrogen_smiles.append(line[0])
        
        if float(line[1]) <= 0:
            negative_smiles.append(line[0])
        else:
            positive_smiles.append(line[0])

data_targets = []
data_preds = []
data_bars = []
data_panel = []

# panel a and b

nitrogen_targets = []
non_nitrogen_targets = []

targets_path = os.path.join(dataset_dir,"dataset","noise20_nitrogen","groupadditivity_test_noise20_nitrogen.csv")
with open(targets_path) as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        if line[0] in nitrogen_smiles:
            nitrogen_targets.append(line[1])
        else:
            non_nitrogen_targets.append(line[1])

preds_path = os.path.join(results_dir,"noise20_nitrogen_mve","0.1","test_preds.csv")
with open(preds_path) as f:
    reader = csv.reader(f)
    next(reader)
    for i, line in enumerate(reader):
        if line[0] in non_nitrogen_smiles[:50]:
            data_targets.append(non_nitrogen_targets[i])
            data_preds.append(line[1])
            data_bars.append(np.sqrt(float(line[2])))
            data_panel.append("a")
        if line[0] in nitrogen_smiles[:50]:
            data_targets.append(nitrogen_targets[i])
            data_preds.append(line[1])
            data_bars.append(np.sqrt(float(line[2])))
            data_panel.append("b")

# panel c and d

negative_targets = []
positive_targets = []

targets_path = os.path.join(dataset_dir,"dataset","noise20_half","groupadditivity_test_noise20_half.csv")
with open(targets_path) as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        if line[0] in negative_smiles:
            negative_targets.append(line[1])
        else:
            positive_targets.append(line[1])

trunc_mve_std = []

preds_path = os.path.join(results_dir,"noise20_half_mve","0.1","test_preds.csv")
with open(preds_path) as f:
    reader = csv.reader(f)
    next(reader)
    for i, line in enumerate(reader):
        if line[0] in negative_smiles[:25]:
            data_targets.append(negative_targets[i])
            data_preds.append(line[1])
            data_bars.append(np.sqrt(float(line[2])))
            trunc_mve_std.append(np.sqrt(float(line[2])))
            data_panel.append("d")
        elif line[0] in positive_smiles[:25]:
            data_targets.append(positive_targets[i])
            data_preds.append(line[1])
            data_bars.append(np.sqrt(float(line[2])))
            trunc_mve_std.append(np.sqrt(float(line[2])))
            data_panel.append("d")

mean_mve_std = np.mean(trunc_mve_std)

trunc_ens_std = []

preds_path = os.path.join(results_dir,"noise20_half","0.1","test_preds.csv")
with open(preds_path) as f:
    reader = csv.reader(f)
    next(reader)
    for i, line in enumerate(reader):
        if line[0] in negative_smiles[:25]:
            data_targets.append(negative_targets[i])
            data_preds.append(line[1])
            trunc_ens_std.append(np.sqrt(float(line[2])))
            data_panel.append("c")
        elif line[0] in positive_smiles[:25]:
            data_targets.append(positive_targets[i])
            data_preds.append(line[1])
            trunc_ens_std.append(np.sqrt(float(line[2])))
            data_panel.append("c")

mean_ens_std = np.mean(trunc_ens_std)
trunc_ens_std = trunc_ens_std/mean_ens_std*mean_mve_std

data_bars.extend(trunc_ens_std.tolist())

write_path = os.path.join(figures_dir, "fig3.csv")
with open(write_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["target","pred","bar","panel"])
    for i in range(len(data_targets)):
        writer.writerow([data_targets[i], data_preds[i], data_bars[i], data_panel[i]])