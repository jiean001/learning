#!/bin/bash -f

# 不同服务器这些设置可能不同
DATA=Capitals_colorGrad64
DATASET="/home/xiongbo/datasets/TOGETHER/${DATA}/"
CHECKPOINTS=/home/xiongbo/results/luxb/reweighted_font_transfer
CUDA_ID=0
GPU_IDS=0
BATCHSIZE=32

experiment_dir="Classifier_pretrain"
MODEL=classifier
NORM=batch
IN_NC=3
O_NC=26
FINESIZE=64
LOADSIZE=64

# train
# NITER=500
# NITERD=100
# EPOCH_FREQ=5
# EMBEDDING_FREQ=50
# BETA1=0.5
# LR=0.0002

# test
RESULT_DIR=/home/xiongbo/test/luxb/reweighted_font_transfer
EMBEDDING_FREQ=1

# continue train
WHICH_EPOCH=15

if [ ! -d "${RESULT_DIR}/${experiment_dir}" ]; then
	mkdir "${RESULT_DIR}/${experiment_dir}"
fi

LOG="${RESULT_DIR}/${experiment_dir}/output.txt"
if [ -f $LOG ]; then
	rm $LOG
fi

exec &> >(tee -a "$LOG")

CUDA_VISIBLE_DEVICES=${CUDA_ID} python ../controller/test_classifier.py --dataroot ${DATASET} --name "${experiment_dir}"\
						 --model $MODEL --which_model_net_Classifier Classifier_letter \
						 --norm $NORM --input_nc $IN_NC --output_nc $O_NC --fineSize $FINESIZE --loadSize $LOADSIZE --use_dropout \
						 --batchSize $BATCHSIZE  \
						 --gpu_ids $GPU_IDS  --classifier --checkpoints_dir $CHECKPOINTS \
						 --use_tensorboardX --embedding_freq $EMBEDDING_FREQ \
                         --which_epoch $WHICH_EPOCH \
                         --results_dir $RESULT_DIR