#!/bin/bash -f

# 不同服务器这些设置可能不同
DATA=Capitals_colorGrad64
DATASET="/home/xiongbo/datasets/SEPARATE/${DATA}/"
CHECKPOINTS=/home/xiongbo/results/luxb/reweighted_font_transfer
CUDA_ID=0
GPU_IDS=0
BATCHSIZE=2
NTHREAD=1
test_type=$1

NGF=64
experiment_dir="reweighted_l_train_ngf64_0725"
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

# loader classifer
LOADER_CLASSIFIER_EPOCH=60
LOADER_CLASSIFIER_NAME=Classifier
# 0:style, 1:content
WHICH_NET_LOADER_CLASSIFIER=0
S_C_CONFIG=style_classifier.txt
C_C_CONFIG=content_classifier.txt


# continue train
WHICH_EPOCH=20

CONSTANT_COS=1

# test
RESULT_DIR=/home/xiongbo/test/luxb/reweighted_font_transfer

if [ ! -d "${RESULT_DIR}/${experiment_dir}" ]; then
	mkdir "${RESULT_DIR}/${experiment_dir}"
fi

LOG="${RESULT_DIR}/${experiment_dir}/output.txt"
if [ -f $LOG ]; then
	rm $LOG
fi

exec &> >(tee -a "$LOG")

CUDA_VISIBLE_DEVICES=${CUDA_ID} python ../controller/test_rew_gan.py --dataroot ${DATASET} --name "${experiment_dir}"\
						 --model $MODEL --which_model_net_Classifier Classifier_letter --which_model_netG $MODEL_NETG\
						 --norm $NORM --input_nc $IN_NC --output_nc $O_NC --fineSize $FINESIZE --loadSize $LOADSIZE --use_dropout \
						 --batchSize $BATCHSIZE  \
						 --gpu_ids $GPU_IDS  --reweighted --checkpoints_dir $CHECKPOINTS \
						 --use_tensorboardX --embedding_freq $EMBEDDING_FREQ \
						 --nThreads $NTHREAD \
						 --ftX_comment $COMMENT \
						 --loader_classifier_epoch $LOADER_CLASSIFIER_EPOCH \
						 --loader_classifier_name $LOADER_CLASSIFIER_NAME \
						 --which_net_loader_classifier $WHICH_NET_LOADER_CLASSIFIER \
						 --s_c_config $S_C_CONFIG \
						 --c_c_config $C_C_CONFIG \
						 --ngf $NGF \
						 --constant_cos $CONSTANT_COS \
						 --results_dir $RESULT_DIR \
						 --config_dir $CONFIG_DIR \
                         --which_epoch $WHICH_EPOCH \
                         --postConv --test_type $test_type\
