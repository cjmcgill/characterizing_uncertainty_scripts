#!/usr/bin/python3

import csv
import numpy as np
import os
from rdkit import Chem
import pandas as pd

results_dir="results"  # indicate the location of the directory where you will be storing your trained models
dataset_dir="data"  # indicate the location of the splits data directory, provided with this github repo. You will need to unzip it before use.
figures_dir = os.path.join(results_dir, "figures")

def chemprop_ensemble(preds_paths):
    for i,path in enumerate(preds_paths):
        preds = []
        smiles = []
        with open(path) as f:
            reader=csv.reader(f)
            header = next(reader)
            for line in reader:
                smiles.append(line[0])
                preds.append(line[1])
        preds = np.array(preds, dtype=float)
        if i == 0:
            sum_preds = preds
        else:
            sum_preds = sum_preds + preds
    preds = sum_preds / len(preds_paths)
    return header, smiles, preds


for split in ["correct","random"]:
    for seed in [0,1,2,3,4]:
        preds_paths = []
        preds_path = os.path.join(results_dir,f"save_u_{split}_100000_{seed}","test_preds.csv")
        preds_paths.append(preds_path)
    header, smiles, preds = chemprop_ensemble(preds_paths)
    write_path = os.path.join(figures_dir,f"u_{split}_100000_preds.csv")
    with open(write_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(smiles)):
            writer.writerow([smiles[i], preds[i]])