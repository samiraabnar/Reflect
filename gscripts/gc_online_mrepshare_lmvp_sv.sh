#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2 \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr8 \
--student_exp_name=gc_std8 \
--teacher_config=big_gpt_v5 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_3 > run_sv_lmvp13 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr9 \
--student_exp_name=gc_std9 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp14 &

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2 \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr10 \
--student_exp_name=gc_std10 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_3 > run_sv_lmvp15 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2 \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr11 \
--student_exp_name=gc_std11 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_2 > run_sv_lmvp16 &



wait