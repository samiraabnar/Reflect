#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=lm_sst2 \
--student_task=sst2  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr1 \
--student_exp_name=gc_std1 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_sst_8_2 > run_sv_lmsst1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=lm_sst2 \
--student_task=sst2  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr2 \
--student_exp_name=gc_std2 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_sst_8_2 > run_sv_lmsst2 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=lm_sst2 \
--student_task=sst2  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr3 \
--student_exp_name=gc_std3 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_sst_8_2 > run_sv_lmsst3 &


CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--teacher_task=lm_sst2 \
--student_task=sst2  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr4 \
--student_exp_name=gc_std4 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_sst_6_2 > run_sv_lmsst4 &

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--teacher_task=lm_sst2 \
--student_task=sst2  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr115 \
--student_exp_name=gc_std115 \
--teacher_config=big_gpt_v5 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_sst_6_2 > run_sv_lmsst5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--teacher_task=lm_sst2 \
--student_task=sst2  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr116 \
--student_exp_name=gc_std116 \
--teacher_config=big_gpt_v5 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_sst_6_2 > run_sv_lmsst6 &

wait