#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_tchr1 \
--student_exp_name=gc_std1 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lmvp1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_tchr2 \
--student_exp_name=gc_std2 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_4 > run_sv_lmvp2 &

#CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
#--teacher_task=word_sv_agreement_lm \
#--student_task=word_sv_agreement_vp  \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=cl_gpt2_shared \
#--teacher_exp_name=gc_mor_dtchr1 \
#--student_exp_name=gc_std1 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=big_gpt_v5 \
#--batch_size=64 \
#--distill_mode=mrep_online \
#--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lmvp3 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
#--teacher_task=word_sv_agreement_lm \
#--student_task=word_sv_agreement_vp  \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=cl_gpt2_shared \
#--teacher_exp_name=gc_mor_dtchr2 \
#--student_exp_name=gc_std2 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=big_gpt_v5 \
#--batch_size=64 \
#--distill_mode=mrep_online \
#--distill_config=rpdst_019_crs_slwfst_4 > run_sv_lmvp4 &



wait