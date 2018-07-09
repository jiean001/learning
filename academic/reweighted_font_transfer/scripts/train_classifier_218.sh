#!/bin/bash -f

# 不同服务器这些设置可能不同
DATA=Capitals_colorGrad64
DATASET="/home/share/dataset/MCGAN/TOGETHER/${DATA}/"
CHECKPOINTS=/home/luxb/results/luxb/reweighted_font_transfer
CUDA_ID=1,0,2
GPU_IDS=0,1,2
BATCHSIZE=256

experiment_dir="Classifier_pretrain"
MODEL=classifier
NORM=batch
IN_NC=3
O_NC=26
FINESIZE=64
LOADSIZE=64
NITER=500
NITERD=100

EPOCH_FREQ=5
EMBEDDING_FREQ=50
BETA1=0.5
LR=0.0002

# continue train
WHICH_EPOCH=20

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
						 --batchSize $BATCHSIZE  \
						 --gpu_ids $GPU_IDS  --classifier --checkpoints_dir $CHECKPOINTS \
						 --use_tensorboardX --embedding_freq $EMBEDDING_FREQ \

                         --which_epoch $WHICH_EPOCH \
                         --beta1 $BETA1 --lr $LR \
						 --save_epoch_freq $EPOCH_FREQ --niter $NITER --niter_decay $NITERD
						 # --serial_batches