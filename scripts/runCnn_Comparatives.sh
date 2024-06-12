export TF_CPP_MIN_LOG_LEVEL=2

# miniTest="mini"
miniTest="big"

lossCurves="" # "-loss_curves True"


if [[ $miniTest == "mini" ]]
then
    ## Mini Tests parameters
    batch_size=32
    epochs=1
    partitions=10
    # datasets="-dbs 1" 
    datasets="-multiple_datasets True"
    pos=1
    neg=5    
    samples="-samples 100" # Not all. Some exp must be with only one always
else
    ## Decent Tests parameters
    batch_size=32
    epochs=200 
    partitions=10
    pos=1
    neg=5
    datasets="-multiple_datasets True"
    samples="-multiple_set_samples True" # Not all. Some exp must be with only one always
fi


common_params_uno_uno_new_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets True ${loss_curves} -pretrained_weights imagenet ${samples} -pos ${pos} -neg ${neg})

# CLean previous executions
#rm -f ./log/saveLosses/*


#### Balance data distribution and batch 
# New Scratch --> Init.W + F.Layers
# --- New Scratch + balance.Batch
expName="cnn"
# expName="scratch"


###Pruebas
# trainSize=20
# limitTrainSize=${trainSize}
# ####
# multSamples=1
# sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
# log_file=log/terminalOutputs/${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} 








trainSize=200
limitTrainSize=${trainSize}
###
# TrainSize 200
multSamples=1
sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}



# Multiply samples per 2
multSamples=2
sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}




trainSize=300
limitTrainSize=${trainSize}
###
# TrainSize 300
multSamples=1
sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}



# Multiply samples per 3
multSamples=3
sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}




trainSize=100
limitTrainSize=${trainSize}
####
## TrainSize 400
#multSamples=1
#sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
#log_file=log/terminalOutputs/${sufix}.txt
#echo "Writing to " ${log_file}
#${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}



# Multiply samples per 3
multSamples=1
sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}










trainSize=400
limitTrainSize=${trainSize}
####
## TrainSize 400
#multSamples=1
#sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
#log_file=log/terminalOutputs/${sufix}.txt
#echo "Writing to " ${log_file}
#${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}



# Multiply samples per 3
multSamples=4
sufix="cnn-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-MultSamples"${multSamples}"-Exp6"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} > ${log_file}




