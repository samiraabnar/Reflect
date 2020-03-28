#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_tchr3 \
--student_exp_name=gc_std3 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_3 > run_sv_lmvp5 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_tchr4 \
--student_exp_name=gc_std4 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_2 > run_sv_lmvp6 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_gpt2_shared \
--teacher_exp_name=gc_mor_dtchr6 \
--student_exp_name=gc_std6 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_3 > run_sv_lmvp7 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_gpt2_shared \
--teacher_exp_name=gc_mor_dtchr5 \
--student_exp_name=gc_std5 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_2 > run_sv_lmvp8 &

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr3 \
--student_exp_name=gc_std3 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_3 > run_sv_lmvp9 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr4 \
--student_exp_name=gc_std4 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_019_crs_slwfst_2 > run_sv_lmvp10 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr6 \
--student_exp_name=gc_std6 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lmvp11 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr7 \
--student_exp_name=gc_std7 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lmvp12 &

wait