#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm  \
--student_task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--teacher_exp_name=gc_or_tchr49 \
--student_exp_name=gc_std49 \
--teacher_config=lstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lm1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm  \
--student_task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--teacher_exp_name=gc_or_tchr48 \
--student_exp_name=gc_std48 \
--teacher_config=biglstm_drop31_v3 \
--student_config=biglstm_drop31_v3 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lm2 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm  \
--student_task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--teacher_exp_name=gc_or_tchr47 \
--student_exp_name=gc_std47 \
--teacher_config=biglstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lm3 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--student_task=word_sv_agreement_lm  \
--teacher_task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--teacher_exp_name=gc_mor_tchr46 \
--student_exp_name=gc_std46 \
--teacher_config=biglstm_drop31_v3 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lm4 &


CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr38 \
--student_exp_name=gc_std38 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr37 \
--student_exp_name=gc_std37 \
--teacher_config=biglstm_drop31_v3 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr36 \
--student_exp_name=gc_std36 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm7 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_tchr35 \
--student_exp_name=gc_std35 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_5 > run_sv_lm8 &

wait