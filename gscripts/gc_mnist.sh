#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist

#
#CUDA_VISIBLE_DEVICES=0 python mnist_trainer.py \
#--model=resnet \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=svhn_adam_mid \
#--batch_size=128 \
#--exp_name=v1 > run1 &
#
#
#CUDA_VISIBLE_DEVICES=1 python mnist_trainer.py \
#--model=resnet \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=svhn_radam_mid \
#--batch_size=128 \
#--exp_name=v2 > run2 &
#
#CUDA_VISIBLE_DEVICES=2 python mnist_trainer.py \
#--model=resnet \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=svhn_crs_slw \
#--batch_size=128 \
#--exp_name=v3 > run3 &
#
#CUDA_VISIBLE_DEVICES=3 python mnist_trainer.py \
#--model=resnet \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=crs_fst \
#--batch_size=128 \
#--exp_name=v4 > run4 &
#
#CUDA_VISIBLE_DEVICES=4 python mnist_trainer.py \
#--model=resnet \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=crs_slw \
#--batch_size=128 \
#--exp_name=v5 > run4 &
#
#CUDA_VISIBLE_DEVICES=5 python mnist_trainer.py \
#--model=cl_vff \
#--task=svhn \
#--model_config=ff_mnist \
#--train_config=svhn_crs_slw \
#--batch_size=128 \
#--exp_name=v6 > run6 &
#
#CUDA_VISIBLE_DEVICES=6 python mnist_trainer.py \
#--model=cl_vff \
#--task=svhn \
#--model_config=ff_mnist \
#--train_config=svhn_adam_mid \
#--batch_size=128 \
#--exp_name=v7 > run7 &
#
#CUDA_VISIBLE_DEVICES=7 python mnist_trainer.py \
#--model=cl_vff \
#--task=svhn \
#--model_config=ff_mnist \
#--train_config=svhn_radam_mid \
#--batch_size=128 \
#--exp_name=v8 > run7 &

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py \
--task=mnist40 \
--teacher_model=resnet \
--student_model=cl_vff \
--student_exp_name=gc_o_std31 \
--teacher_exp_name=gc_o_dtchr31 \
--teacher_config=rsnt_mnist1 \
--student_config=ff_mnist1 \
--distill_mode=online \
--batch_size=128 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &


CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py \
--task=mnist40 \
--teacher_model=resnet \
--student_model=cl_vff \
--student_exp_name=gc_o_std32 \
--teacher_exp_name=gc_o_dtchr32 \
--teacher_config=rsnt_mnist1 \
--student_config=ff_mnist2 \
--distill_mode=online \
--batch_size=128 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run2 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
--task=mnist40 \
--teacher_model=resnet \
--student_model=cl_vff \
--student_exp_name=gc_o_std33 \
--teacher_exp_name=gc_o_dtchr33 \
--teacher_config=rsnt_mnist1 \
--student_config=ff_mnist3 \
--distill_mode=online \
--batch_size=128 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run3 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py \
--task=mnist40 \
--teacher_model=resnet \
--student_model=cl_vff \
--student_exp_name=gc_o_std34 \
--teacher_exp_name=gc_o_dtchr34 \
--teacher_config=rsnt_mnist3 \
--student_config=ff_mnist1 \
--distill_mode=online \
--batch_size=128 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run4 &

CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py \
--task=mnist40 \
--teacher_model=resnet \
--student_model=cl_vff \
--student_exp_name=gc_o_std35 \
--teacher_exp_name=gc_o_dtchr35 \
--teacher_config=rsnt_mnist3 \
--student_config=ff_mnist2 \
--distill_mode=online \
--batch_size=128 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run5 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py \
--task=mnist40 \
--teacher_model=resnet \
--student_model=cl_vff \
--student_exp_name=gc_o_std36 \
--teacher_exp_name=gc_o_dtchr36 \
--teacher_config=rsnt_mnist3 \
--student_config=ff_mnist3 \
--distill_mode=online \
--batch_size=128 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run6 &


wait