#!/bin/bash 
#SBATCH --array=0-19

# Script for generating the data needed for Figures 7 and 8.
# Training and inference time are reduced significantly when using a gpu to train.
# Training is parallelizable across the individual jobs, shown here as jobs in a slurm job array.
# Jobs must be run with the appropriate conda environment for chemprop activated.


chemprop_dir=chemprop  # indicate the location of the chemprop directory on your local computer
results_dir=results  # indicate the location of the directory where you will be storing your trained models
dataset_dir=groupadditivity_h298  # indicate the location of the group additivity dataset directory
seed=$SLURM_ARRAY_TASK_ID

# Base

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/groupadditivity_0.1.csv \
--save_dir $results_dir/ensemble_base/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/ensemble_base/${seed}  \
--preds_path $results_dir/ensemble_base/preds_${seed}.csv \
# --gpu 0


# Dataset sizes

for fraction in 0.001 0.0001; do

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/groupadditivity_${fraction}.csv \
--save_dir $results_dir/ensemble_fraction${fraction}/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/ensemble_fraction${fraction}/${seed}  \
--preds_path $results_dir/ensemble_fraction${fraction}/preds_${seed}.csv \
# --gpu 0

done

# Hidden sizes

for size in 20 100; do

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/groupadditivity_0.1.csv \
--save_dir $results_dir/ensemble_size${size}/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size $size \
--ffn_hidden_size $size \
--epochs 200 \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/ensemble_size${size}/${seed}  \
--preds_path $results_dir/ensemble_size${size}/preds_${seed}.csv \
# --gpu 0

done

# Training Noise Level

for noise in 0.02 1; do

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/noise${noise}/groupadditivity_0.1_noise${noise}.csv \
--save_dir $results_dir/ensemble_noise${noise}/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/ensemble_noise${noise}/${seed}  \
--preds_path $results_dir/ensemble_noise${noise}/preds_${seed}.csv \
# --gpu 0

done

# Aggregation Mean

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/groupadditivity_0.1.csv \
--save_dir $results_dir/ensemble_aggreagation_mean/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation mean \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/ensemble_aggregation_mean/${seed}  \
--preds_path $results_dir/ensemble_aggregation_mean/preds_${seed}.csv \
# --gpu 0


