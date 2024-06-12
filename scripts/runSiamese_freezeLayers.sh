export TF_CPP_MIN_LOG_LEVEL=2

# miniTest="mini"
miniTest="big"

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
    datasets="-dbs 1"
    samples="-multiple_set_samples True" # Not all. Some exp must be with only one always
fi

# Tests


## Test individual tests

# CLean previous executions
#rm -f ./log/saveLosses/*

## Check efectivity of freezing layers when init weights
## Scratch (Imbalanced scenario) vs Init (imbalanced scenario)
common_params_uno_uno=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets True -neg 1 -pos 1 -loss_curves True ${reload_data})
common_params_init_w_dsALL=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} -neg 1 -pos 1 -loss_curves True -pretrained_weights imagenet ${reload_data})




# --- Scratch + Init weights + Freeze all layers
expName=original_Imbalanced_distribution-batch_Imbalance
freezeLayers="ALL"
log_file=log/terminalOutputs/${expName}-init_weights-freeze${freezeLayers}.txt
echo "Writing to " ${log_file}
${common_params_init_w_dsALL[@]} ${samples} -freeze ${freezeLayers} -type ${expName} -log_table ${expName}-init_weights-freeze${freezeLayers} > ${log_file}

# --- Scratch  + Init weights (No frozening)
expName="original_Imbalanced_distribution-batch_Imbalance"
log_file=log/terminalOutputs/${expName}-init_weights.txt
echo "Writing to " ${log_file}
${common_params_uno_uno[@]} ${samples} -pretrained_weights imagenet -type ${expName} -log_table ${expName}-init_weights > ${log_file} 
