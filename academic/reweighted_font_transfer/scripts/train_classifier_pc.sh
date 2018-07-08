#!/bin/bash -f

DATA=A
DATASET="/home/xiongbo/datasets/ft6_1/${DATA}/"
experiment_dir="Classifier_pretrain"

MODEL=classifier
NORM=batch
IN_NC=3
O_NC=26

FINESIZE=64
LOADSIZE=64
NITER=500
NITERD=100
BATCHSIZE=64
CUDA_ID=0
EPOCH_FREQ=2
CHECKPOINTS=/home/xiongbo/results/luxb/reweighted_font_transfer
EMBEDDING_FREQ=5

if [ ! -d "${CHECKPOINTS}/${experiment_dir}" ]; then
	mkdir "${CHECKPOINTS}/${experiment_dir}"
fi
LOG="${CHECKPOINTS}/${experiment_dir}/output.txt"
if [ -f $LOG ]; then
	rm $LOG
fi

exec &> >(tee -a "$LOG")

CUDA_VISIBLE_DEVICES=${CUDA_ID} python ../controller/train_classifier.py --dataroot ${DATASET} --name "${experiment_dir}"\
						 --model $MODEL --which_model_net_Classifier Classifier_letter \
						 --norm $NORM --input_nc $IN_NC --output_nc $O_NC --fineSize $FINESIZE --loadSize $LOADSIZE --use_dropout \
						 --niter $NITER --niter_decay $NITERD --batchSize $BATCHSIZE --save_epoch_freq $EPOCH_FREQ \
						 --gpu_ids 0 --which_epoch 40 --classifier --checkpoints_dir $CHECKPOINTS \
						 --use_tensorboardX --serial_batches --embedding_freq $EMBEDDING_FREQ















