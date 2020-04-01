#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr111 \
--student_exp_name=gc_std111 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp111 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr112 \
--student_exp_name=gc_std112 \
--teacher_config=very_big_gpt_v10 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp112 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr113 \
--student_exp_name=gc_std113 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_2 > run_sv_lmvp113 &


CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr114 \
--student_exp_name=gc_std114 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_3 > run_sv_lmvp114 &

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr115 \
--student_exp_name=gc_std115 \
--teacher_config=very_big_gpt_v10 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_3 > run_sv_lmvp115 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr116 \
--student_exp_name=gc_std116 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_3 > run_sv_lmvp116 &

wait