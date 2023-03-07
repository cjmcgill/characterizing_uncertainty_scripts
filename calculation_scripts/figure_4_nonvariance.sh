#!/bin/sh
#SBATCH --array=0-146

# Script for calculating the nonvariance errors in Figure 4
# Generating the entire data set can be very time intensive, so reducing the script to produce a subset may be more appropriate.
# GPU resources are not useful for this task.
# Training is parallelizable across the individual jobs, shown here as jobs in a slurm job array.
# Jobs must be run with the appropriate conda environment for chemprop activated.


ensemble_projection_dir=ensemble_projection  # indicate the location of the ensemble_projection repository on your local computer
results_dir=results  # indicate the location of the directory where the trained models and test set predictions are stored
dataset_dir=groupadditivity_h298  # indicate the location of the group additivity dataset directory
# Script also assumes an assigned $TMPDIR directory variable

fraction_array=({1e-05,2e-05,4e-05,6e-05,8e-05,0.0001,0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04,0.06,0.08,0.1})
size_array=({20,50,100,200,1000,1500,2000})
fraction=${fraction_array[$(($SLURM_ARRAY_TASK_ID / 7))]}
size=${size_array[$(($SLURM_ARRAY_TASK_ID % 7))]}

python $ensemble_projection_dir/main.py \
--target_path $dataset_dir/dataset/groupadditivity_test.csv \
--preds_path $results_dir/size${size}/${fraction}/test_preds.csv \
--ensemble_size 5 \
--save_dir $results_dir/size${size}/${fraction} \
--bw_multiplier 0.5 \
--convergence_method kl_threshold \
--kl_threshold 2.5e-6 \
--scratch_dir $TMPDIR/$SLURM_ARRAY_TASK_ID \
--likelihood_calculation calculated \
--max_projection_size 5
