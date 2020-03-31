#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr108 \
--student_exp_name=gc_std108 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp108 &


CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr109 \
--student_exp_name=gc_std109 \
--teacher_config=very_big_gpt_v10 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp109 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr105 \
--student_exp_name=gc_std105 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_2 > run_sv_lmvp110 &

wait