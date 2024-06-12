#!/bin/bash

gpu=0

DBS="cifar10 mnist fashion_mnist usps"
#MODE=pair   # 'pair', 'pairclass', 'triplet', 'tripletclass', 'tripletmem'




# TRAINING size... BEST CONFIG WITH TEST AUGMENTATION
for db in $DBS; do
	for size in 1 5 10 20 50 75 100 200 500; do
		python -u experimenter.py -db $db -mode pairclass -size $size -augtr 6 --augte --knn -gpu $gpu > out_BEST_pairclass_${db}_size_${size}_augtr_6_augte.txt
	done
done



# CNN...
for db in $DBS; do
	python -u experimenter.py -db $db -mode cnn -gpu $gpu > out_CNN_${db}.txt
done


# OPTIMIZERS...
for db in $DBS; do
	for opt in sgd rmsprop adadelta adam nadam; do
		python -u experimenter.py -db $db -mode pair -opt $opt -gpu $gpu > out_pair_${db}_opt_${opt}.txt
	done
done


# DIMENSIONALITY...
for db in $DBS; do
	for dim in 2 4 8 16 32 64 128 256 512 1024; do
		python -u experimenter.py -db $db -mode pair -dim $dim -gpu $gpu > out_pair_${db}_dim_${dim}.txt
	done
done


# LOSS...
for db in $DBS; do
	for loss in a b c; do
		python -u experimenter.py -db $db -mode pair -loss $loss -gpu $gpu > out_pair_${db}_loss_${loss}.txt
	done
done


# MARGIN...
for db in $DBS; do
	for margin in 0.25 0.5 1.0 2.0 4.0 8.0; do
		python -u experimenter.py -db $db -mode pair -margin $margin -gpu $gpu > out_pair_${db}_margin_${margin}.txt
	done
done


# CLASS LOSS CONTRIBUTION...
for db in $DBS; do
	for contrib in 0.063 0.125 0.25 0.5 1.0 2.0 4.0; do
		python -u experimenter.py -db $db -mode pairclass -contrib $contrib -gpu $gpu > out_pairclass_${db}_contrib_${contrib}.txt
	done
done


# PAIRING PROPORTION...
POSITIVE="4 3 2 3 4 1 3 2 1 1 1 1 1 1 1 1 1"
NEGATIVE=(1 1 1 2 3 1 4 3 2 3 4 5 6 7 8 9 10)
POS=0

for db in $DBS; do
	POS=0
	for p in $POSITIVE; do
 		python -u experimenter.py -db $db -mode pair -pos $p -neg ${NEGATIVE[$POS]} -gpu $gpu > out_pair_${db}_pos_${p}_neg_${NEGATIVE[$POS]}.txt
		((POS++))
	done
done


# TRAINING AUGMENTATION...
for db in $DBS; do
	for aug in 2 4 6 8 10 20; do
		python -u experimenter.py -db $db -mode pair -augtr $aug -gpu $gpu > out_pair_${db}_augtr_${aug}.txt
	done
done


# TRAINING size... FOR CNN...!!!
for db in $DBS; do
	for size in 1 5 10 20 50 75 100 200 500; do
		python -u experimenter.py -db $db -mode cnn -size $size -gpu $gpu > out_CNN_${db}_size_${size}.txt
	done
done


# EVALUATE TOPOLOGIES...
for db in $DBS; do
	for top in pair pairclass triplet tripletclass tripletmem; do
		python -u experimenter.py -db $db -mode $top -gpu $gpu > out_TOP_${top}_${db}.txt
	done
done


# TRAINING size...
for db in $DBS; do
	for size in 1 5 10 20 50 75 100 200 500; do
		python -u experimenter.py -db $db -mode pair -size $size -gpu $gpu > out_pair_${db}_size_${size}.txt
	done
done


# TRAINING size... BEST CONFIG
for db in $DBS; do
	for size in 1 5 10 20 50 75 100 200 500; do
		python -u experimenter.py -db $db -mode pairclass -size $size -augtr 6 --knn -gpu $gpu > out_BEST_pairclass_${db}_size_${size}.txt
	done
done


# TEST AUGMENTATION...
for db in $DBS; do
	python -u experimenter.py -db $db -mode pairclass -augtr 6 --augte -gpu $gpu > out_pairclass_${db}_augtr_6_augte.txt
	python -u experimenter.py -db $db -mode pairclass --augte -gpu $gpu > out_pairclass_${db}_augte.txt
done




