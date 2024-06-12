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
    reload_data=""
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
# common_params_uno_uno=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets True -neg 1 -pos 1 ${loss_curves} ${reload_data})
# common_params_uno_uno_tl_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${loss_curves} ${samples})
common_params_uno_uno_new_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${loss_curves} ${samples} ${datasets})


# CLean previous executions
#rm -f ./log/saveLosses/*

## Check efectivity of freezing layers when init weights
## Scratch (Imbalanced scenario) vs Init (imbalanced scenario)
# common_params_init_w_dsALL=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} -neg 1 -pos 1 ${loss_curves} -pretrained_weights imagenet ${reload_data})


## BEST CASE FOR INFERENCE MODEL
#### Balance data distribution and batch 
# New Scratch --> Init.W + F.Layers
# --- New Scratch + balance.Batch
# expName="transfer-learning"
# pos=5
# neg=1
# sufix="Bal-TL-pos"${pos}"_neg"${neg}"-Exp4"
# log_file=log/terminalOutputs/${sufix}.txt
# rm -f log/resultsTable/${sufix}.txt
#
# # Orden de pre-trained datasets en tabla. 9 exps
# init_dbs=(2 3 2 3 1 1 2 1 1)
# dbs=(1 1 1 2 2 2 3 3 3)
# tgt_dbs=(1 2 3 1 2 3 1 2 3)
# echo "Writing to " ${log_file}
# for (( exp=0; exp<${#init_dbs[@]}; exp++ ))
# do
#     echo "Exp "${exp}", pre-trained "${init_dbs[$exp]}", trained ds "${dbs[$exp]}", target ds"${tgt_dbs[$exp]}
#     ${common_params_uno_uno_tl_scratch[@]} -pos ${pos} -neg ${neg} -init_dbs ${init_dbs[$exp]} -dbs ${dbs[$exp]} -tl_based_tgt_dbs ${tgt_dbs[$exp]} -type ${expName} -log_table ${sufix} #> ${log_file}
# done




expName="scratch"
pos=5
neg=1
sufix="Bal-InitW-pos"${pos}"_neg"${neg}"-Exp4"
log_file=log/terminalOutputs/${sufix}.txt
rm -f log/resultsTable/${sufix}.txt
${common_params_uno_uno_new_scratch[@]} -pos ${pos} -neg ${neg} -type ${expName} -log_table ${sufix} > ${log_file}


pos=3
neg=2
sufix="Bal-InitW-pos"${pos}"_neg"${neg}"-Exp4"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -pos ${pos} -neg ${neg} -type ${expName} -log_table ${sufix} > ${log_file}


pos=2
neg=3
sufix="Bal-InitW-pos"${pos}"_neg"${neg}"-Exp4"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -pos ${pos} -neg ${neg} -type ${expName} -log_table ${sufix} > ${log_file}


pos=1
neg=5
sufix="Bal-InitW-pos"${pos}"_neg"${neg}"-Exp4"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -pos ${pos} -neg ${neg} -type ${expName} -log_table ${sufix} > ${log_file}

