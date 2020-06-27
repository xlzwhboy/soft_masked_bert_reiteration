#!/bin/bash

TASKNAME=sighan_all

# train
OUTPUT_D_PATH=../../Model/${TASKNAME}

# bert init path
INIT_MODEL_D_PATH="../../Model/chinese_L-12_H-768_A-12"
VOCAB_F_PATH=${INIT_MODEL_D_PATH}"/vocab.txt"
MODEL_CHECKPOINT_PATH=${INIT_MODEL_D_PATH}"/bert_model.ckpt"
CONFIG_F_PATH=${INIT_MODEL_D_PATH}/bert_config.json

# train params
BATCH_SIZE=64

if [[ $1 == 'train_data' ]]; then
  python make_tfrecord.py --vocab_file=${VOCAB_F_PATH} \
  --output_dir=${OUTPUT_D_PATH}

elif [[ $1 == 'train' ]]; then
  python train.py \
  --task_name=${TASKNAME} \
  --init_checkpoint=${MODEL_CHECKPOINT_PATH} \
  --bert_config_file=${CONFIG_F_PATH} \
  --input_file=${OUTPUT_D_PATH}/train_128.tf_record \
  --output_dir=${OUTPUT_D_PATH} \
  "${@:2}"

else

  echo 'unknown argment 1, must be train_data|eval_data|train|evaluate'
fi
