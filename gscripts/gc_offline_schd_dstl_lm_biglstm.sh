#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2 LSTM

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_tmp5_rep_std1 \
--teacher_exp_name=gc_tmp5_rep_tchr1 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=off_schdld \
--distill_config=schdexp_dstl5_910_crs_slwfst_2 > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_tmp1_rep_std10 \
--teacher_exp_name=gc_tmp1_rep_tchr10 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=off_schdld \
--distill_config=schdexp_dstl_910_crs_slwfst_2 > run1 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_tmp01_rep_std20 \
--teacher_exp_name=gc_tmp01_rep_tchr20 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=off_schdld \
--distill_config=schdexp_dstl01_910_crs_slwfst_2 > run2 &

# LSTM 2 Transformer

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_tmp5_rep_std1 \
--teacher_exp_name=gc_tmp5_rep_tchr1 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=off_schdld \
--distill_config=schdexp_dstl5_910_crs_slwfst_2 > run3 &

CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_tmp1_rep_std10 \
--teacher_exp_name=gc_tmp1_rep_tchr10 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=off_schdld \
--distill_config=schdexp_dstl_910_crs_slwfst_2 > run4 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_tmp01_rep_std20 \
--teacher_exp_name=gc_tmp01_rep_tchr20 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=off_schdld \
--distill_config=schdexp_dstl01_910_crs_slwfst_2  > run5&

wait
