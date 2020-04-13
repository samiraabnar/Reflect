#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn3 \
--train_config=svhn_adam_mid \
--batch_size=128 \
--exp_name=v25 > run1 &


CUDA_VISIBLE_DEVICES=1 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn3 \
--train_config=svhn_radam_mid \
--batch_size=128 \
--exp_name=v26> run2 &

CUDA_VISIBLE_DEVICES=2 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn3 \
--train_config=svhn_crs_slw \
--batch_size=128 \
--exp_name=v27 > run3 &

CUDA_VISIBLE_DEVICES=3 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn3 \
--train_config=crs_fst \
--batch_size=128 \
--exp_name=v28 > run4 &

CUDA_VISIBLE_DEVICES=4 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn4 \
--train_config=svhn_radam_mid \
--batch_size=128 \
--exp_name=v29 > run5 &

CUDA_VISIBLE_DEVICES=5 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn4 \
--train_config=svhn_adam_mid \
--batch_size=128 \
--exp_name=v30 > run6 &

CUDA_VISIBLE_DEVICES=6 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn4 \
--train_config=crs_fst \
--batch_size=128 \
--exp_name=v31 > run7 &

CUDA_VISIBLE_DEVICES=7 python mnist_trainer.py \
--model=resnet \
--task=svhn \
--model_config=rsnt_svhn4 \
--train_config=svhn_crs_slw \
--batch_size=128 \
--exp_name=v32 > run8 &

wait