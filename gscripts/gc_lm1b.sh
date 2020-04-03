#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python keras_trainer.py \
--model=lm_lstm_shared_emb \
--task=lm1b \
--model_config=lstm_drop31_v3 \
--train_config=radam_slw2 \
--batch_size=512 \
--exp_name=offlineteacher_v1 > lm1b_run1 &


wait