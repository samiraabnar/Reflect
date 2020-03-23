#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--teacher_exp_name=gc_or_dtchr3 \
--student_exp_name=gc_std3 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst3 > run_sst8 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--teacher_exp_name=gc_or_dtchr4 \
--student_exp_name=gc_std4 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst4 > run_sst9 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py --task=sst2  \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--teacher_exp_name=gc_or_dtchr5 \
--student_exp_name=gc_std5 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=128 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_sst5 > run_sst10 &


wait