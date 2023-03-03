# Script for generating the data needed for Figures 5, 6, 10, and 11.
# Generating the entire data set can be very time intensive, so reducing the script to produce a subset may be more appropriate.
# Training and inference time are reduced significantly when using a gpu to train.
# Training is parallelizable across the individual jobs.
# Jobs must be run with the appropriate conda environment activated.



chemprop_dir=chemprop  # indicate the location of the chemprop directory on your local computer
results_dir=results  # indicate the location of the directory where you will be storing your trained models
dataset_dir=data  # indicate the location of the splits data directory, provided with this github repo
spk_dir=schnet  # indicate the location of the schnetpack directory
qm9_path=qm9.db  # path to qm9.db as provided with this github repo. You will need to unzip it before use.

# MPNN Enthalpy and Gap, norm and mean aggregation
for i in 100 300 1000 3000 10000 30000 100000; do
    for agg in norm mean; do
	for prop in enthalpy_H gap; do
	    python $chemprop_dir/train.py \
		   --data_path $dataset_dir/train_${i}.csv \
		   --separate_test_path $dataset_dir/test.csv\
		   --separate_val_path $dataset_dir/val_${i}.csv\
		   --dataset_type regression \
		   --num_workers 20 \
		   --metric mae \
		   --ensemble_size 5\
		   --aggregation ${agg} \
		   --target_columns ${prop}\
		   --depth 4 \
		   --ffn_num_layers 2 \
		   --save_dir $results_dir/save_${prop}_${agg}_${i} \
		   --epochs	200
	done
    done
done

#FFN Enthalpy and Gap
for i in 100 300 1000 3000 10000 30000 100000; do
    for fp in 10 100 1000; do
        for prop in gap enthalpy_H; do
            python $chemprop_dir/train.py \
                   --data_path $dataset_dir/train_${i}.csv \
                   --separate_test_path $dataset_dir/test.csv\
                   --separate_val_path $dataset_dir/val_${i}.csv\
                   --features_path $dataset_dir/train_fp${fp}_${i}.csv \
                   --separate_test_features_path $dataset_dir/test_fp${fp}.csv\
                   --separate_val_features_path $dataset_dir/val_fp${fp}_${i}.csv \
                   --features_only\
                   --dataset_type regression \
                   --num_workers 20 \
                   --metric mae \
                   --ensemble_size 5\
                   --target_columns ${prop}\
                   --ffn_num_layers 2 \
                   --save_dir $results_dir/save_${prop}_fp${fp}_${i} \
                   --epochs     200
            done
    done
done

#Schnet #use spk_run.py from the Schnetpack package
for i in 100 300 1000 3000 10000 30000 100000; do
    for agg in sum avg; do
        for prop in gap enthalpy_H; do
            for s in 0 1 2 3 4; do
                mkdir $results_dir/save_schnet_${prop}_${agg}_${i}_${s}
                cp $dataset_dir/split_${i}_${prop}.npz $results_dir/save_schnet_${prop}_${agg}_${i}_${s}/split.npz
                python $spk_dir/spk_run.py train schnet custom $qm9_path $results_dir/save_schnet_${prop}_${agg}_${i}_${s} --split_path split.npz --cuda --property ${prop} --aggregation_mode ${agg} --seed ${s}
                python $spk_dir/spk_run.py eval $qm9_path $results_dir/save_schnet_${prop}_${agg}_${i}_${s} --split test --cuda
		python $spk_dir/spk_run.py pred $dataset_dir/qm9.db $results_dir/save_schnet_${prop}_${agg}_${i}_${s} --split test
            done
        done
    done
done

#MPNN on different molecular sizes
for i in 100 300 1000 3000 10000 30000 100000; do
    for agg in norm mean; do
	python $chemprop_dir/train.py \
            --data_path $dataset_dir/train_agg_size${i}.csv \
            --separate_test_path $dataset_dir/test_agg_size_small.csv\
            --separate_val_path $dataset_dir/val_agg_size${i}.csv\
            --dataset_type regression \
            --num_workers 20 \
            --metric mae \
            --ensemble_size 5\
            --aggregation ${agg} \
            --target_columns enthalpy_H\
            --depth 4 \
            --ffn_num_layers 2 \
            --save_dir $results_dir/save_agg_${agg}_size_${i} \
            --epochs     200

    python $chemprop_dir/predict.py \
           --checkpoint_dir $results_dir/save_agg_${agg}_size_${i} \
           --test_path $dataset_dir/test_agg_size_small.csv\
           --preds_path $results_dir/save_agg_${agg}_size_${i}/preds_small.csv
    
    python $chemprop_dir/predict.py \
           --checkpoint_dir $results_dir/save_agg_${agg}_size_${i} \
           --test_path $dataset_dir/test_agg_size_large.csv\
           --preds_path $results_dir/save_agg_${agg}_size_${i}/preds_large.csv
    done
done

# MPNN on U at different temperatures
for i in 100 300 1000 3000 10000 30000 100000; do
    for s in correct random; do
            python $chemprop_dir/train.py \
                   --data_path $dataset_dir/u_${s}_train_${i}.csv \
                   --separate_test_path $dataset_dir/u_${s}_test.csv\
                   --separate_val_path $dataset_dir/u_${s}_val_${i}.csv\
                   --features_path $dataset_dir/u_${s}_train_features_${i}.csv \
                   --separate_test_features_path $dataset_dir/u_${s}_test_features.csv\
                   --separate_val_features_path $dataset_dir/u_${s}_val_features_${i}.csv \
                   --dataset_type regression \
                   --num_workers 20 \
                   --metric mae \
                   --ensemble_size 5\
                   --aggregation norm\
                   --target_columns u_atom\
                   --depth 4\
                   --ffn_num_layers 2 \
                   --save_dir $results_dir/save_u_${s}_${i} \
                   --epochs     200
    done
done
