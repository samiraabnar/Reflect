#!/usr/bin/env bash


export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr5 \
--student_exp_name=gc_std5 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst6 > run_sst5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr6 \
--student_exp_name=gc_std6 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_00199_crs_slwfst_sst6 > run_sst6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--teacher_exp_name=gc_or_tchr7 \
--student_exp_name=gc_std7 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst5 > run_sst7 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--teacher_exp_name=gc_or_dtchr1 \
--student_exp_name=gc_std1 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst1 > run_sst8 &

wait