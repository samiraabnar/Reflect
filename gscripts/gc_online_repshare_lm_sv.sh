#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr74 \
--student_exp_name=gc_std74 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr75 \
--student_exp_name=gc_std75 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm2 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr76 \
--student_exp_name=gc_std76 \
--teacher_config=biglstm_drop31_v3 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm3 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr77 \
--student_exp_name=gc_std77 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm4 &


CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr78 \
--student_exp_name=gc_std78 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr79 \
--student_exp_name=gc_std79 \
--teacher_config=biglstm_drop31_v3 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr80 \
--student_exp_name=gc_std80 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_6 > run_sv_lm7 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr81 \
--student_exp_name=gc_std81 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_7 > run_sv_lm8 &

wait