#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py --task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--teacher_exp_name=gc_or_tchr1 \
--student_exp_name=gc_std1 \
--teacher_config=lstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm1 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py --task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr1 \
--student_exp_name=gc_std1 \
--teacher_config=very_big_gpt_v10 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm2 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py --task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr2 \
--student_exp_name=gc_std2 \
--teacher_config=very_big_gpt_v10 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lm3 &

wait