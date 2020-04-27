#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2 LSTM

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std1 \
--teacher_exp_name=gc_o_tchr1 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > o_run0 &


CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std2 \
--teacher_exp_name=gc_o_tchr2 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst > o_run0 &


#LSTM to Transformer
CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std3 \
--teacher_exp_name=gc_o_tchr3 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > o_run0 &


CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std4 \
--teacher_exp_name=gc_o_tchr4 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst > o_run0 &


#Transformer to Transformer
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std5 \
--teacher_exp_name=gc_o_tchr5 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > o_run0 &


CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std6 \
--teacher_exp_name=gc_o_tchr6 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst > o_run0 &


wait
