export TF_CPP_MIN_LOG_LEVEL=2

# miniTest="mini"
miniTest="big"

lossCurves="" # "-loss_curves True"


if [[ $miniTest == "mini" ]]
then
    ## Mini Tests parameters
    batch_size=32
    epochs=1
    partitions=1
    datasets="-dbs 3" 
    samples="-samples 100"
    reload_data="-reload_data True"
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
# common_params_uno_uno=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets True -neg 1 -pos 1 ${loss_curves} ${reload_data})
# dbs_init=1
# fileWeights=${dbs_init}__samples100__pos-1__neg-1__e200__b32__dim32__experimentscratch-Imb_Dist-Imb_Batch-Exp1__size100__freeze0__partition0
common_params_uno_uno_from_tl_1=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets True -neg 1 -pos 1 ${loss_curves} ${reload_data} -pretrained_weights imagenet)

# CLean previous executions
#rm -f ./log/saveLosses/*


#### Augmentations
# --- Balanced dist + balanced.Batch + Augmentations
sufix="Imb-InitW-Exp3"
expName="data_aug"

# augm=1
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
# log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} #> ${log_file}
#
#
# augm=5
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
# log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} > ${log_file}

augm=10
rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} > ${log_file}

augm=15
rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} > ${log_file}

# augm=20
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
# log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} > ${log_file}
#
# augm=30
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
# log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} > ${log_file}
#
# augm=40
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}${sufix})
# log_file=log/terminalOutputs/${expName}${augm}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_from_tl_1[@]} ${rest_params[@]} > ${log_file}






## Test big table with and without augmentation
# scratch, Siamese-transfer-learning
# experiments=("init_weights" "data_aug" "data_aug" "data_aug" "weight_loss" "scratch")
# experiments=("init_weights")
# # Augmentation parameter per each experiment
# # augtrs=(0 1 2 3 0 0)
# augtrs=(0)
# for (( exp=0; exp<${#augtrs[@]}; exp++ ))
# do
#     expNameNeg=${experiments[$exp]}"variaNegs_"${exp}
#     expNameSamples=${experiments[$exp]}"variaSamples_"${exp}
#     common_params=( python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -augtr ${augtrs[$exp]} -multiple_datasets "True")
#     variaSamples=(-neg 5 -pos 1  -multiple_set_samples "True" -type ${experiments[$exp]} -log_table ${expNameSamples})
#     variaNegs=(-multiple_neg "True" ${samples} -type ${experiments[$exp]} -log_table ${expNameNeg})
#     ${common_params[@]} ${variaSamples[@]} > "log/terminalOutputs/"${expNameSamples}".txt"
#     ${common_params[@]} ${variaNegs[@]} > "log/terminalOutputs/"${expNameNeg}".txt"
# done


# # 
## Save weights of all augment vs no augment
# Aug
# python3 main_launch_experiments.py -e 200 -b 8 -neg 5 -dbs 1 -samples 5 -type "Siamese-data-aug" -save_weights "True"
# python3 main_launch_experiments.py -e 200 -b 8 -neg 5 -dbs 2 -samples 5 -type "Siamese-data-aug" -save_weights "True"
# python3 main_launch_experiments.py -e 200 -b 8 -neg 5 -dbs 3 -samples 5 -type "Siamese-data-aug" -save_weights "True"
# # No aug
# python3 main_launch_experiments.py -e 200 -b 8 -neg 5 -dbs 1 -samples 5 -type "init_weights" -save_weights "True"
# python3 main_launch_experiments.py -e 200 -b 8 -neg 5 -dbs 2 -samples 5 -type "init_weights" -save_weights "True"
# python3 main_launch_experiments.py -e 200 -b 8 -neg 5 -dbs 3 -samples 5 -type "init_weights" -save_weights "True"


## Important arguments
# -multiple_datasets: Test on all 3 datasets
# -multiple_set_samples: Test experiment on set of sample images in training minority class for each dataset
# --multiple_neg: Test a set of negative pairings
