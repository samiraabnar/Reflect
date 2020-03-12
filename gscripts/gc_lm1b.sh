#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=biglstm_drop31_v2 \
--train_config=radam_slw \
--exp_name=offlineteacher_v1 &


CUDA_VISIBLE_DEVICES=1 python $CODE_DIR/keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=biglstm_drop31_v2 \
--train_config=radam_fst \
--exp_name=offlineteacher_v2 &


CUDA_VISIBLE_DEVICES=2 python $CODE_DIR/keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=biglstm_drop31_v2 \
--train_config=crs_slw \
--exp_name=offlineteacher_v3 &

wait