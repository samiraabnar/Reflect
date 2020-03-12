#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=biglstm_drop31_v2 \
--train_config=radam_slw \
--exp_name=offlineteacher_v1 > lm1b_run1 &


CUDA_VISIBLE_DEVICES=1 python keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=biglstm_drop31_v2 \
--train_config=radam_fst \
--exp_name=offlineteacher_v2 > lm1b_run2&


CUDA_VISIBLE_DEVICES=2 python keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=biglstm_drop31_v2 \
--train_config=crs_slw \
--exp_name=offlineteacher_v3 > lm1b_run3&

wait