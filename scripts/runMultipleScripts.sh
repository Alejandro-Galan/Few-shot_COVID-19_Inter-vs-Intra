#!/bin/bash


 
# Transfer Learning initw and frozen layers with 10 partitions
# echo "Experiment 1.2 File: Transfer Learning with frozen layers"
# ./scripts/runSiamese_TransferLearning.sh


#
# # Repeat Exp1 with 10 partitions
# echo "Experiment 1 File: Initialization"
# ./scripts/runSiamese_Initialization.sh
#
# # # Exp2 over TL 10 partitions with InitW training weights
# echo "Experiment 2 File: Balance and InitW over TL"
# ./scripts/runSiamese_ImbalancedExps.sh




# # Exp3 over TL 10 partitions with InitW Data Augmentation
# echo "Experiment 3 File: Data Augmentation"
# ./scripts/runSiamese_DataAugmentation.sh


# Exp5 over TL 10 partitions with InitW Train Size
echo "Experiment 5 File: Train Size"
./scripts/runSiamese_TrainTotalSize.sh


# Exp4 over TL 10 partitions with InitW Pairings
echo "Experiment 4 File: Pairings"
./scripts/runSiamese_PairingNegs.sh

# Exp6 over TL 10 partitions with InitW Train Size
# echo "Experiment 6 File: cnn over Train Size"
# ./scripts/runCnn_Comparatives.sh





