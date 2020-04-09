#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm  \
--student_task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--teacher_exp_name=gc_or_tchr42 \
--student_exp_name=gc_std42 \
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
--teacher_exp_name=gc_or_tchr43 \
--student_exp_name=gc_std43 \
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
--teacher_exp_name=gc_or_tchr44 \
--student_exp_name=gc_std44 \
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
--teacher_exp_name=gc_mor_tchr45 \
--student_exp_name=gc_std45 \
--teacher_config=biglstm_drop31_v3 \
--student_config=lstm_drop31_v2 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_019_crs_slwfst_3 > run_sv_lm4 &


CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr31 \
--student_exp_name=gc_std31 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr32 \
--student_exp_name=gc_std32 \
--teacher_config=biglstm_drop31_v3 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr28 \
--student_exp_name=gc_std28 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm7 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2_shared \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_tchr27 \
--student_exp_name=gc_std27 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_2 > run_sv_lm8 &

wait