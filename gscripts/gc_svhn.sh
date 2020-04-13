#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=svhn_adam_mid \
--batch_size=128 \
--exp_name=v33 > run1 &


CUDA_VISIBLE_DEVICES=1 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=svhn_radam_mid \
--batch_size=128 \
--exp_name=v34 > run2 &

CUDA_VISIBLE_DEVICES=2 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn1 \
--train_config=svhn_crs_slw \
--batch_size=128 \
--exp_name=v35 > run3 &

CUDA_VISIBLE_DEVICES=3 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn5 \
--train_config=svhn_crs_slw \
--batch_size=128 \
--exp_name=v36 > run4 &

CUDA_VISIBLE_DEVICES=4 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn5 \
--train_config=svhn_radam_mid \
--batch_size=128 \
--exp_name=v37 > run5 &

CUDA_VISIBLE_DEVICES=5 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn5 \
--train_config=svhn_adam_mid \
--batch_size=128 \
--exp_name=v38 > run6 &

CUDA_VISIBLE_DEVICES=6 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn5 \
--train_config=adam_mid \
--batch_size=128 \
--exp_name=v39 > run7 &

CUDA_VISIBLE_DEVICES=7 python mnist_trainer.py \
--model=cl_vcnn \
--task=svhn \
--model_config=vcnn_svhn5 \
--train_config=radam_mid \
--batch_size=128 \
--exp_name=v40 > run8 &

wait