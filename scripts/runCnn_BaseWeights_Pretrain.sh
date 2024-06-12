export TF_CPP_MIN_LOG_LEVEL=2

miniTest="mini"
# miniTest="big"

lossCurves="" # "-loss_curves True"


if [[ $miniTest == "mini" ]]
then
    ## Mini Tests parameters
    batch_size=32
    epochs=1
    partitions=10
    datasets="-dbs 1" 
    # datasets="-multiple_datasets True"
    pos=1
    neg=1
    samples="-samples -1" # Not all. Some exp must be with only one always
else
    ## Decent Tests parameters
    batch_size=32
    epochs=50
    partitions=1
    pos=1
    neg=1
    samples="-samples -1" # Not all. Some exp must be with only one always
fi


common_params_uno_uno_new_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${loss_curves} -pretrained_weights imagenet ${samples} -pos ${pos} -neg ${neg})

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








# dbs=1
# trainSize=-1
# limitTrainSize=${trainSize}
# multSamples=1
# sufix="cnn_tl_dbs"${dbs}"-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-SAVE_WEIGHTS"
# log_file=log/terminalOutputs/${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} -dbs ${dbs} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} #> ${log_file}


dbs=2
trainSize=-1
limitTrainSize=${trainSize}
multSamples=1
sufix="cnn_tl_dbs"${dbs}"-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-SAVE_WEIGHTS"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}
${common_params_uno_uno_new_scratch[@]} -dbs ${dbs} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} #> ${log_file}

# dbs=3
# trainSize=-1
# limitTrainSize=${trainSize}
# multSamples=1
# sufix="cnn_tl_dbs"${dbs}"-Bal-InitW-pos"${pos}"_neg"${neg}"-trainSize"${trainSize}"-SAVE_WEIGHTS"
# log_file=log/terminalOutputs/${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} -dbs ${dbs} -multSamples ${multSamples} -limit_train_size ${limitTrainSize} -train_size ${trainSize} -type ${expName} -log_table ${sufix} #> ${log_file}




