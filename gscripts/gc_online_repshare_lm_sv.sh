#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr82 \
--student_exp_name=gc_std82 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr83 \
--student_exp_name=gc_std83 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm2 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr84 \
--student_exp_name=gc_std84 \
--teacher_config=biglstm_drop31_v3 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm3 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr85 \
--student_exp_name=gc_std85 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm4 &


CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr86 \
--student_exp_name=gc_std86 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr87 \
--student_exp_name=gc_std87 \
--teacher_config=biglstm_drop31_v3 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr89 \
--student_exp_name=gc_std89 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm7 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr90 \
--student_exp_name=gc_std90 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_25 > run_sv_lm8 &

wait