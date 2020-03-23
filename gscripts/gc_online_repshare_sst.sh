#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr1 \
--student_exp_name=gc_std1 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr2 \
--student_exp_name=gc_std2 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst2 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr3 \
--student_exp_name=gc_std3 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst3 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr4 \
--student_exp_name=gc_std4 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst4 &


CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr5 \
--student_exp_name=gc_std5 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst5 &

wait