#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2 LSTM

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std7 \
--teacher_exp_name=gc_o_tch7 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst3 > o_run0 &


CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std8 \
--teacher_exp_name=gc_o_tchr8 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst2 > o_run0 &


#LSTM to Transformer
CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std9 \
--teacher_exp_name=gc_o_tchr9 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst4 > o_run0 &


CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std10 \
--teacher_exp_name=gc_o_tchr10 \
--teacher_config=biglstm_drop31_v2 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst3 > o_run0 &


#Transformer to Transformer
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std11 \
--teacher_exp_name=gc_o_tchr11 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst > o_run0 &


CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--student_exp_name=gc_o_std12 \
--teacher_exp_name=gc_o_tchr12 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run0 &


wait
