export TF_CPP_MIN_LOG_LEVEL=2

# miniTest="mini"
miniTest="big"

lossCurves="" #"-loss_curves True"

if [[ $miniTest == "mini" ]]
then
    ## Mini Tests parameters
    batch_size=32
    epochs=1
    partitions=1
    samples="-samples 100"
    reload_data="" #"-reload_data False"
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
# common_params_uno_cinco=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} -multiple_datasets "True" -neg 5 -pos 1 ${lossCurves})
# common_params_uno_uno=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} -neg 1 -pos 1 ${lossCurves} ${reload_data})
# common_params_uno_uno_new_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} -neg 1 -pos 1 ${lossCurves} ${reload_data} -pretrained_weights imagenet)

# common_params_uno_uno_tl_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions}  ${lossCurves} ${reload_data} ${samples})
common_params_uno_uno_initW_scratch=(python3 main_launch_experiments.py -e ${epochs} -b ${batch_size} -cv_partitions ${partitions} ${datasets} ${lossCurves} ${reload_data} ${samples} -pretrained_weights imagenet)

# CLean previous executions
#rm -f ./log/saveLosses/*


# # # --- Scratch
# expName="original_Imbalanced_distribution-batch_Imbalance"
# log_file=log/terminalOutputs/${expName}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${samples} -type ${expName} -log_table ${expName} > ${log_file}

# # --- Scratch  + Init weights (No frozening)
# expName="original_Imbalanced_distribution-batch_Imbalance"
# log_file=log/terminalOutputs/${expName}-init_weights.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${samples} -pretrained_weights imagenet -type ${expName} -log_table ${expName}-init_weights > ${log_file} 

# --- Scratch + Init weights + Freeze all layers
# expName=original_Imbalanced_distribution-batch_Imbalance
# sufix="-init_weights-FrozL"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} ${samples} -dist_batch_imbalance True -type ${expName} -log_table ${expName}${sufix} > ${log_file}


#### Balance data distribution and batch 
# New Scratch --> Init.W 

########################################
########################################
#### SCRATCH BASE --> Transf Learn  ####
########################################
########################################


# --- New Scratch + Balanced.Data
expName="transfer-learning"
sufix="Bal-InitW-Exp2"
log_file=log/terminalOutputs/${sufix}.txt
rm -f log/resultsTable/${sufix}.txt

echo "Writing to " ${log_file}
echo "Exp "${exp}", pre-trained "${init_dbs[$exp]}", trained ds "${dbs[$exp]}", target ds"${tgt_dbs[$exp]}
${common_params_uno_uno_initW_scratch[@]} -neg 1 -pos 1 -type ${expName} -log_table ${sufix} > ${log_file}

#### W.loss

# W.loss over original distributions
# --- New Scratch + W.Loss
sufix="Imb-WLoss-InitW-Exp2"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}

${common_params_uno_uno_initW_scratch[@]} -dist_batch_imbalance True -w_loss True -type ${expName} -log_table ${sufix} > ${log_file}


# W.loss over balanced distributions
# --- New Scratch + Balanced.Data + W.Loss
expName="transfer-learning"
sufix="Bal-WLoss-InitW-Exp2"
log_file=log/terminalOutputs/${sufix}.txt
echo "Writing to " ${log_file}

${common_params_uno_uno_initW_scratch[@]} -w_loss True -neg 1 -pos 1 -type ${expName} -log_table ${sufix} > ${log_file}












# ########################################
# ########################################
# #### SCRATCH BASE --> Transf Learn  ####
# ########################################
# ########################################
#
#
# # --- New Scratch + Balanced.Data
# expName="transfer-learning"
# sufix="Bal-TL-Exp2"
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
#     ${common_params_uno_uno_tl_scratch[@]} -init_dbs ${init_dbs[$exp]} -dbs ${dbs[$exp]} -tl_based_tgt_dbs ${tgt_dbs[$exp]} -neg 1 -pos 1 -type ${expName} -log_table ${sufix} > ${log_file}
# done
#
# #### W.loss
#
# # W.loss over original distributions
# # --- New Scratch + W.Loss
# sufix="Imb-WLoss-TL-Exp2"
# log_file=log/terminalOutputs/${sufix}.txt
# echo "Writing to " ${log_file}
# for (( exp=0; exp<${#init_dbs[@]}; exp++ ))
# do
#     echo "Exp "${exp}", pre-trained "${init_dbs[$exp]}", trained ds "${dbs[$exp]}", target ds"${tgt_dbs[$exp]}
#     ${common_params_uno_uno_tl_scratch[@]} -dist_batch_imbalance True -w_loss True -init_dbs ${init_dbs[$exp]} -dbs ${dbs[$exp]} -tl_based_tgt_dbs ${tgt_dbs[$exp]} -type ${expName} -log_table ${sufix} > ${log_file}
# done
#
# # W.loss over balanced distributions
# # --- New Scratch + Balanced.Data + W.Loss
# expName="transfer-learning"
# sufix="Bal-WLoss-TL-Exp2"
# log_file=log/terminalOutputs/${sufix}.txt
# echo "Writing to " ${log_file}
# for (( exp=0; exp<${#init_dbs[@]}; exp++ ))
# do
#     echo "Exp "${exp}", pre-trained "${init_dbs[$exp]}", trained ds "${dbs[$exp]}", target ds"${tgt_dbs[$exp]}
#     ${common_params_uno_uno_tl_scratch[@]} -w_loss True -init_dbs ${init_dbs[$exp]} -dbs ${dbs[$exp]} -tl_based_tgt_dbs ${tgt_dbs[$exp]} -neg 1 -pos 1 -type ${expName} -log_table ${sufix} > ${log_file}
# done


#############################################
#############################################
#### CHANGE OF BASE SCRATCH --> F.LAYERS ####
#############################################
#############################################

# # --- New Scratch + Balanced.Data
# expName="scratch"
# sufix="-Bal_Dist-Bal_Batch-FLayers-Exp2"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} -freeze ALL  ${samples} -type ${expName} -log_table ${expName}${sufix} > ${log_file} 


# #### W.loss

# # W.loss over original distributions
# # --- New Scratch + W.Loss 
# expName="scratch"
# sufix="-Imb_Dist-Imb_Batch-WLoss-FLayers-Exp2"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} -freeze ALL -dist_batch_imbalance True ${samples} -w_loss True -type ${expName} -log_table ${expName}${sufix} > ${log_file}

# # W.loss over original distributions
# # --- New Scratch + Balanced.Data + W.Loss 
# expName="scratch"
# sufix="-Bal_Dist-Bal_Batch-WLoss-FLayers-Exp2"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]} -freeze ALL  ${samples} -w_loss True -type ${expName} -log_table ${expName}${sufix} > ${log_file}



###################################################
###################################################
#### CHANGE OF BASE SCRATCH --> TL -> ChestGit ####
###################################################
###################################################

# # --- New Scratch + Balanced.Data
# expName="transfer-learning"
# sufix="-Bal-TL_ChestGit-Exp2"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_tl_scratch[@]} -tl_weights ${fileWeights} -init_dbs ${dbs_init} ${samples} -type ${expName} -log_table ${expName}${sufix} > ${log_file} 


# #### W.loss

# # W.loss over original distributions
# # --- New Scratch + W.Loss 
# expName="transfer-learning"
# sufix="-Imb-WLoss-TL_ChestGit-Exp2"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_tl_scratch[@]} -tl_weights ${fileWeights} -init_dbs ${dbs_init} -dist_batch_imbalance True ${samples} -w_loss True -type ${expName} -log_table ${expName}${sufix} > ${log_file}

# # W.loss over original distributions
# # --- New Scratch + Balanced.Data + W.Loss 
# expName="transfer-learning"
# sufix="-Bal-WLoss-TL_ChestGit-Exp2"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_tl_scratch[@]} -tl_weights ${fileWeights} -init_dbs ${dbs_init} ${samples} -w_loss True -type ${expName} -log_table ${expName}${sufix} > ${log_file}



###################################################
###################################################



# # --- New Scratch + balance.Batch + W.loss 
# expName="original_Imbalanced_distribution"
# sufix="-batch_Balanced-w_loss-initW-FrozL"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]}  ${samples} -w_loss "True" -type ${expName} -log_table ${expName}${sufix} > ${log_file}

# # --- New Scratch + Balanced dist + balanced.Batch + W.loss
# expName="weight_loss"
# sufix="-balanced_distribution-batch_Balanced-initW-FrozL"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]}  ${samples} -w_loss "True" -type ${expName}  -log_table ${expName}${sufix}  > ${log_file}


# # --- New Scratch + Balanced dist + balanced.Batch + Init.W + Frozen Layers + W.loss
# expName="weight_loss"
# sufix="-balanced_distribution-batch_Balanced-init_W_FrozenLayers-initW-FrozL"
# log_file=log/terminalOutputs/${expName}${sufix}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno_new_scratch[@]}  ${samples} -w_loss "True" -freeze ${freezeLayers} -pretrained_weights imagenet -type ${expName} -log_table ${expName}${sufix}  > ${log_file}


# #### Augmentations
# # --- Balanced dist + balanced.Batch + Augmentations
# expName="data_aug_low"
# log_file=log/terminalOutputs/${expName}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${samples} -augtr 1 -type ${expName} -log_table ${expName}_balanced_data-batch_Balanced > ${log_file}

# expName="data_aug_high"
# augm=10
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}_balanced_data-batch_Balanced)
# log_file=log/terminalOutputs/${expName}${augm}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${rest_params[@]} > ${log_file}

# augm=20
# expName="data_aug_high"
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}_balanced_data-batch_Balanced)
# log_file=log/terminalOutputs/${expName}${augm}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${rest_params[@]} > ${log_file}

# augm=30
# expName="data_aug_high"
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}_balanced_data-batch_Balanced)
# log_file=log/terminalOutputs/${expName}${augm}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${rest_params[@]} > ${log_file}

# augm=40
# expName="data_aug_high"
# rest_params=(${samples} -augtr ${augm} -type ${expName} -log_table ${expName}${augm}_balanced_data-batch_Balanced)
# log_file=log/terminalOutputs/${expName}${augm}.txt
# echo "Writing to " ${log_file}
# ${common_params_uno_uno[@]} ${rest_params[@]} > ${log_file}






## Test big table with and without augmentation
# scratch, Siamese-transfer-learning
# experiments=("init_weights" "data_aug_low" "data_aug_high" "data_aug_high" "weight_loss" "scratch")
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
