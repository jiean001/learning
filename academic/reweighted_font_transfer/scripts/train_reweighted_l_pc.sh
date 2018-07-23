#!/bin/bash -f

# 不同服务器这些设置可能不同
DATA=Capitals_colorGrad64
DATASET="/home/xiongbo/datasets/SEPARATE/${DATA}/"
CHECKPOINTS=/home/xiongbo/results/luxb/reweighted_font_transfer
CUDA_ID=0
GPU_IDS=0
BATCHSIZE=4
NTHREAD=1

experiment_dir="reweighted_l_train"
MODEL=reweighted_l
MODEL_NETG=reweighted_gan
COMMENT=reweighted_l
NORM=batch
IN_NC=3
O_NC=4
FINESIZE=64
LOADSIZE=64
CONFIG_DIR=../config/

# train
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

CUDA_VISIBLE_DEVICES=${CUDA_ID} python ../controller/train_rew_gan.py --dataroot ${DATASET} --name "${experiment_dir}"\
						 --model $MODEL --which_model_net_Classifier Classifier_letter --which_model_netG $MODEL_NETG\
						 --norm $NORM --input_nc $IN_NC --output_nc $O_NC --fineSize $FINESIZE --loadSize $LOADSIZE --use_dropout \
						 --batchSize $BATCHSIZE  \
						 --gpu_ids $GPU_IDS  --reweighted --checkpoints_dir $CHECKPOINTS \
						 --use_tensorboardX --embedding_freq $EMBEDDING_FREQ \
						 --nThreads $NTHREAD \
						 --ftX_comment $COMMENT \
						 --isTrain \
						 --config_dir $CONFIG_DIR \
                         --which_epoch $WHICH_EPOCH \
                         --beta1 $BETA1 --lr $LR \
                         --postConv \
						 --save_epoch_freq $EPOCH_FREQ --niter $NITER --niter_decay $NITERD
						 # --serial_batches
