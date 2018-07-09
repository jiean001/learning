#!/bin/bash -f

DATA=Capitals_colorGrad64
DATASET="/home/share/dataset/MCGAN/TOGETHER/${DATA}/"
experiment_dir="Classifier_pretrain"

MODEL=classifier
NORM=batch
IN_NC=3
O_NC=26

FINESIZE=64
LOADSIZE=64
NITER=500
NITERD=100
BATCHSIZE=256
CUDA_ID=1,0
EPOCH_FREQ=5
CHECKPOINTS=/home/luxb/results/luxb/reweighted_font_transfer
# CHECKPOINTS=/home/xiongbo/results/luxb/reweighted_font_transfer
EMBEDDING_FREQ=8

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
						 --gpu_ids 0,1 --which_epoch 40 --classifier --checkpoints_dir $CHECKPOINTS \
						 --use_tensorboardX --embedding_freq $EMBEDDING_FREQ \
						 # --serial_batches














