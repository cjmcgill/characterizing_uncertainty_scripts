#!/bin/bash 
#SBATCH --array=0-104

# Script for generating the data needed for Figure 1a. Some redundancy with the data needed for other figures.
# Generating the entire data set can be very time intensive, so reducing the script to produce a subset may be more appropriate.
# Training and inference time are reduced significantly when using a gpu to train.
# Training is parallelizable across the individual jobs, shown here as jobs in a slurm job array.
# Jobs must be run with the appropriate conda environment for chemprop activated.



chemprop_dir=chemprop  # indicate the location of the chemprop directory on your local computer
results_dir=results  # indicate the location of the directory where you will be storing your trained models
dataset_dir=groupadditivity_h298  # indicate the location of the group additivity dataset directory

noise=1
fraction_array=({1e-05,2e-05,4e-05,6e-05,8e-05,0.0001,0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04,0.06,0.08,0.1})
seed_array=({1,2,3,4,5})
fraction=${fraction_array[$(($SLURM_ARRAY_TASK_ID / 5))]}
seed=${seed_array[$(($SLURM_ARRAY_TASK_ID % 5))]}

# noisy training

python $chemprop_dir/train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/noise${noise}/groupadditivity_${fraction}_noise${noise}.csv \
--save_dir $results_dir/noise${noise}/${fraction}/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0
 
python $chemprop_dir/predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/noise${noise}/${fraction}/${seed} \
--preds_path $results_dir/noise${noise}/${fraction}/${seed}/test_preds.csv \
# --gpu 0 

python $chemprop_dir/predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/noise${noise}/${fraction} \
--preds_path $results_dir/noise${noise}/${fraction}/test_preds.csv \
# --gpu 0 


# clean training

python $chemprop_dir/train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/groupadditivity_${fraction}.csv \
--save_dir $results_dir/size1000/${fraction}/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0
 
python $chemprop_dir/predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/size1000/${fraction}/${seed} \
--preds_path $results_dir/size1000/${fraction}/${seed}/test_preds.csv \
# --gpu 0 

python $chemprop_dir/predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/size1000/${fraction} \
--preds_path $results_dir/size1000/${fraction}/test_preds.csv \
--ensemble_variance \
# --gpu 0 


