#!/bin/bash 
#SBATCH --array=0-29

# Script for generating the data needed for Figure 9. Some redundancy with the data needed for Figure 4.
# Training and inference time are reduced significantly when using a gpu to train.
# Training is parallelizable across the individual jobs, shown here as jobs in a slurm job array.
# Jobs must be run with the appropriate conda environment for chemprop activated.



chemprop_dir=chemprop  # indicate the location of the chemprop directory on your local computer
results_dir=results  # indicate the location of the directory where you will be storing your trained models
dataset_dir=groupadditivity_h298  # indicate the location of the group additivity dataset directory

fraction=0.1
seed=$SLURM_ARRAY_TASK_ID

# crossvalidation

python train.py \
--dataset_type regression \
--data_path $dataset_dir/dataset/groupadditivity_${fraction}.csv \
--save_dir $results_dir/folds_comparison/cv/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
--extra_metrics mae \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/folds_comparison/cv/${seed} \
--preds_path $results_dir/folds_comparison/cv/independent_preds_${seed}.csv \
# --gpu 0


# ensemble

if [ $seed = 0 ]
then

python train.py \
--dataset_type regression \
--data_path $dataset_dir/dataset/groupadditivity_${fraction}.csv \
--save_dir $results_dir/folds_comparison/ensemble/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
--save_smiles_splits \
# --gpu 0

else

python train.py \
--dataset_type regression \
--data_path $dataset_dir/dataset/groupadditivity_${fraction}.csv \
--save_dir $results_dir/folds_comparison/ensemble/${seed} \
--seed 0 \
--pytorch_seed ${seed} \
--aggregation norm \
--depth 4 \
--ffn_num_layers 2 \
--hidden_size 1000 \
--ffn_hidden_size 1000 \
--epochs 200 \
# --gpu 0

fi

python predict.py \
--test_path $results_dir/folds_comparison/ensemble/0/fold_0/test_smiles.csv \
--checkpoint_dir $results_dir/folds_comparison/ensemble/${seed} \
--preds_path $results_dir/folds_comparison/ensemble/test_preds_${seed}.csv \
# --gpu 0

python predict.py \
--test_path $dataset_dir/dataset/groupadditivity_test.csv \
--checkpoint_dir $results_dir/folds_comparison/ensemble/${seed} \
--preds_path $results_dir/folds_comparison/ensemble/independent_preds_${seed}.csv \
# --gpu 0

