#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=radam_slw2 \
--batch_size=64 \
--exp_name=v1 > run1 &

CUDA_VISIBLE_DEVICES=1 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=adam_slw \
--batch_size=64 \
--exp_name=v2 > run2 &

CUDA_VISIBLE_DEVICES=2 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=adam_mid \
--batch_size=512 \
--exp_name=v3 > run3 &

CUDA_VISIBLE_DEVICES=3 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=crs_slw \
--batch_size=512 \
--exp_name=v4 > run4 &

CUDA_VISIBLE_DEVICES=4 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn2 \
--train_config=radam_slw2 \
--batch_size=64 \
--exp_name=v5 > run5 &

CUDA_VISIBLE_DEVICES=5 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn2 \
--train_config=adam_slw \
--batch_size=64 \
--exp_name=v6 > run6 &

CUDA_VISIBLE_DEVICES=6 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn2 \
--train_config=adam_mid \
--batch_size=512 \
--exp_name=v7 > run7 &

CUDA_VISIBLE_DEVICES=7 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn2 \
--train_config=crs_slw \
--batch_size=512 \
--exp_name=v8 > run8 &

wait