#!/bin/bash 
#SBATCH --array=0-9
"""
Script for generating the data needed for Figure 3.
Training and inference time are reduced significantly when using a gpu to train.
Training is parallelizable across the individual jobs, shown here as jobs in a slurm job array.
Jobs must be run with the appropriate conda environment for chemprop activated.
"""

chemprop_dir=chemprop  # indicate the location of the chemprop directory on your local computer
results_dir=results  # indicate the location of the directory where you will be storing your trained models
dataset_dir=groupadditivity_h298  # indicate the location of the group additivity dataset directory


size=1000
fraction=0.1
seed_array=({1,2,3,4,5})
noise_array=({half,nitrogen})
noise=${noise_array[$((SLURM_ARRAY_TASK_ID/5))]}
seed=${seed_array[$((SLURM_ARRAY_TASK_ID%5))]}

cd ..

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/noise20_${noise}/gdb11_${fraction}_noise20_${noise}.csv \
--save_dir $results_dir/noise20_${noise}/${fraction}/${seed} \
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
--preds_path $results_dir/noise20_${noise}/${fraction}/${seed}/test_preds.csv \
--checkpoint_dir $results_dir/noise20_${noise}/${fraction}/${seed} \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--preds_path $results_dir/noise20_${noise}/${fraction}/test_preds.csv \
--checkpoint_dir $results_dir/noise20_${noise}/${fraction} \
--ensemble_variance \
# --gpu 0

python train.py \
--dataset_type regression \
--split_sizes 0.8 0.2 0 \
--data_path $dataset_dir/dataset/noise20_${noise}/gdb11_${fraction}_noise20_${noise}.csv \
--save_dir $results_dir/noise20_${noise}_mve/${fraction}/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size $size \
--ffn_hidden_size $size \
--epochs 200 \
--loss_function mve \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--preds_path $results_dir/noise20_${noise}_mve/${fraction}/${seed}/test_preds.csv \
--checkpoint_dir $results_dir/noise20_${noise}_mve/${fraction}/${seed} \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--preds_path $results_dir/noise20_${noise}_mve/${fraction}/test_preds.csv \
--checkpoint_dir $results_dir/noise20_${noise}_mve/${fraction} \
--uncertainty_method mve \
# --gpu 0

