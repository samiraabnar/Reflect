#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist

#
#CUDA_VISIBLE_DEVICES=0 python mnist_trainer.py \
#--model=cl_vcnn \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=svhn_adam_mid \
#--batch_size=128 \
#--exp_name=v1 > run1 &
#
#
#CUDA_VISIBLE_DEVICES=1 python mnist_trainer.py \
#--model=cl_vcnn \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=svhn_radam_mid \
#--batch_size=128 \
#--exp_name=v2 > run2 &
#
#CUDA_VISIBLE_DEVICES=2 python mnist_trainer.py \
#--model=cl_vcnn \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=svhn_crs_slw \
#--batch_size=128 \
#--exp_name=v3 > run3 &
#
#CUDA_VISIBLE_DEVICES=3 python mnist_trainer.py \
#--model=cl_vcnn \
#--task=svhn \
#--model_config=rsnt_svhn1 \
#--train_config=crs_fst \
#--batch_size=128 \
#--exp_name=v4 > run4 &
#
#CUDA_VISIBLE_DEVICES=4 python mnist_trainer.py \
#--model=cl_vcnn \
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

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std100 \
#--teacher_exp_name=gc_o_dtchr100 \
#--teacher_config=rsnt_mnist1 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &
#
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std101 \
#--teacher_exp_name=gc_o_dtchr101 \
#--teacher_config=rsnt_mnist1 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_2 > o_run2 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std102 \
#--teacher_exp_name=gc_o_dtchr102 \
#--teacher_config=rsnt_mnist1 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run3 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std104 \
#--teacher_exp_name=gc_o_dtchr104 \
#--teacher_config=rsnt_mnist3 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dst2_4_crs_slw_3 > o_run4 &
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vff \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std202 \
#--teacher_exp_name=gc_o_tchr202 \
#--teacher_config=ff_mnist4 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run5 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vff \
#--student_model=cl_vcnn \
#--student_exp_name=gc_f_std201 \
#--teacher_exp_name=gc_o_dtchr201 \
#--teacher_config=ff_mnist4 \
#--student_config=rsnt_mnist1 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run6 &


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std110 \
#--teacher_exp_name=gc_o_dtchr100 \
#--teacher_config=rsnt_mnist1 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std111 \
#--teacher_exp_name=gc_o_dtchr101 \
#--teacher_config=rsnt_mnist1 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std112 \
#--teacher_exp_name=gc_o_dtchr102 \
#--teacher_config=rsnt_mnist1 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &
#
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vff \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std210 \
#--teacher_exp_name=gc_o_dtchr210 \
#--teacher_config=ff_mnist4 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vff \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std211 \
#--teacher_exp_name=gc_o_dtchr211 \
#--teacher_config=ff_mnist4 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vff \
#--student_model=cl_vff \
#--student_exp_name=gc_f_std212 \
#--teacher_exp_name=gc_o_dtchr212 \
#--teacher_config=ff_mnist4 \
#--student_config=ff_mnist4 \
#--distill_mode=offline \
#--batch_size=128 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vcnn \
#--student_exp_name=gc_f_std17 \
#--teacher_exp_name=gc_o_tchr17 \
#--teacher_config=vcnn_mnist7 \
#--student_config=vcnn_mnist7 \
#--distill_mode=offline \
#--batch_size=128 \
#--keep_some_checkpoints=True \
#--max_checkpoints=15 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run6 &


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vcnn \
#--student_exp_name=gc_f_std18 \
#--teacher_exp_name=gc_o_tchr18 \
#--teacher_config=vcnn_mnist7 \
#--student_config=vcnn_mnist7 \
#--distill_mode=offline \
#--batch_size=128 \
#--keep_some_checkpoints=True \
#--max_checkpoints=15 \
#--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py \
--task=mnist \
--teacher_model=cl_vcnn \
--student_model=cl_vcnn \
--student_exp_name=gc_f_std16 \
--teacher_exp_name=gc_o_tchr16 \
--teacher_config=vcnn_mnist7 \
--student_config=vcnn_mnist7 \
--distill_mode=offline \
--batch_size=128 \
--keep_some_checkpoints=True \
--max_checkpoints=15 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py \
--task=mnist \
--teacher_model=cl_vcnn \
--student_model=cl_vcnn \
--student_exp_name=gc_f_std15 \
--teacher_exp_name=gc_o_tchr15 \
--teacher_config=vcnn_mnist7 \
--student_config=vcnn_mnist7 \
--distill_mode=offline \
--batch_size=128 \
--keep_some_checkpoints=True \
--max_checkpoints=15 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &


CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
--task=mnist \
--teacher_model=cl_vff \
--student_model=cl_vcnn \
--student_exp_name=gc_f_std102 \
--teacher_exp_name=gc_o_tchr102 \
--teacher_config=ff_mnist4 \
--student_config=vcnn_mnist7 \
--distill_mode=offline \
--batch_size=128 \
--keep_some_checkpoints=True \
--max_checkpoints=15 \
--distill_config=pure_dstl5_4_crs_slw_3 > o_run1 &

#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vff \
#--student_model=cl_vcnn \
#--student_exp_name=gc_o_std103 \
#--teacher_exp_name=gc_o_tchr103 \
#--teacher_config=ff_mnist4 \
#--student_config=vcnn_mnist7 \
#--distill_mode=online \
#--batch_size=128 \
#--keep_some_checkpoints=True \
#--max_checkpoints=15 \
#--distill_config=pure_dstl2_4_crs_slw_3 > o_run1 &
#
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py \
#--task=mnist \
#--teacher_model=cl_vcnn \
#--student_model=cl_vff \
#--student_exp_name=gc_o_std1000 \
#--teacher_exp_name=gc_o_dtchr1000 \
#--teacher_config=vcnn_mnist7 \
#--student_config=ff_mnist4 \
#--distill_mode=online \
#--batch_size=128 \
#--keep_some_checkpoints=True \
#--max_checkpoints=15 \
#--distill_config=pure_dstl2_4_crs_slw_3 > o_run1 &

wait