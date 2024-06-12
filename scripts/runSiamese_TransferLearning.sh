export TF_CPP_MIN_LOG_LEVEL=2

# miniTest="mini"
miniTest="big"

if [[ $miniTest == "mini" ]]
then
    ## Mini Tests parameters
    batch_size=32
    epochs=1
    partitions=10
    datasets="-dbs 1" 
    samples="-multiple_set_samples True"
    reload_data="" #"-reload_data True"
else
    ## Decent Tests parameters
    batch_size=32
    epochs=200 
    partitions=10
    datasets="-multiple_datasets True"
    samples="-multiple_set_samples True" # Not all. Some exp must be with only one always
fi

# Tests


## Test individual tests
# Base case "original_Imbalanced_distribution-batch_Imbalance"
# common_params_uno_cinco=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets "True" -neg 5 -pos 1 -loss_curves "True")
common_params_uno_uno=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} ${reload_data} -dist_batch_imbalance True)

# CLean previous executions
#rm -f ./log/saveLosses/*

## Check efectivity of freezing layers when init weights
## Scratch (Imbalanced scenario) vs Init (imbalanced scenario)
# common_params_init_w_dsALL=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} -neg 1 -pos 1 -loss_curves True -pretrained_weights imagenet ${reload_data})



# # --- Scratch  + Init weights (No frozening)
# expName="original_Imbalanced_distribution-batch_Imbalance"
# log_file=log/terminalOutputs/${expName}-init_weights.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} -save_weights True ${samples} -pretrained_weights imagenet -type ${expName} -log_table ${expName}-init_weights > ${log_file} 

# # --- Scratch + Init weights + Freeze all layers
# expName=original_Imbalanced_distribution-batch_Imbalance
# # log_file=log/terminalOutputs/${expName}-init_weights-freezeALL.txt
# echo "Writing to " ${log_file}
# ${common_params_init_w_dsALL[@]} -save_weights True ${samples} -freeze ALL -type ${expName} -log_table ${expName}-init_weights-freezeALL > ${log_file}

# --- Scratch + Transfer Learning (no initW) --> From InitW only
# DB 1
# expName=transfer-learning
# dbs_init=1
# fileWeights=_${dbs_init}_cnn_tl_samples-1__pos1__neg1__e50__b8__dim32__experimentcnn-Bal-InitW-pos1_neg1-trainSize-1-SAVE_WEIGHTS__size-1__pretrainedWimagenet
# sufix=-Imb-InitW-TL_ChestGit-Exp1.2
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} -tl_weights ${fileWeights} -init_dbs ${dbs_init} ${samples} -type ${expName} -log_table ${expName}${sufix}

# DB 2
# expName=transfer-learning
# dbs_init=2
# fileWeights=_${dbs_init}_cnn_tl_samples-1__pos1__neg1__e50__b8__dim32__experimentcnn-Bal-InitW-pos1_neg1-trainSize-1-SAVE_WEIGHTS__size-1__pretrainedWimagenet
# sufix=-Imb-InitW-TL_PADBIM-Exp1.2
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} -tl_weights ${fileWeights} -init_dbs ${dbs_init} ${samples} -type ${expName} -log_table ${expName}${sufix} #> ${log_file}



# DB 3
expName=transfer-learning
dbs_init=3
fileWeights=_${dbs_init}_cnn_tl_samples-1__pos1__neg1__e50__b8__dim32__experimentcnn-Bal-InitW-pos1_neg1-trainSize-1-SAVE_WEIGHTS__size-1__pretrainedWimagenet
sufix=-Imb-InitW-TL_BIMCV-Exp1.2
log_file=log/terminalOutputs/${expName}${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno[@]} -tl_weights ${fileWeights} -init_dbs ${dbs_init} ${samples} -type ${expName} -log_table ${expName}${sufix} #> ${log_file}






## Important arguments
# -multiple_datasets: Test on all 3 datasets
# -multiple_set_samples: Test experiment on set of sample images in training minority class for each dataset
# --multiple_neg: Test a set of negative pairings
