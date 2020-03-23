#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--teacher_exp_name=gc_or_dtchr1 \
--student_exp_name=gc_std1 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst1 > run_sst6 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--teacher_exp_name=gc_or_dtchr2 \
--student_exp_name=gc_std2 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst2 > run_sst7 &



wait