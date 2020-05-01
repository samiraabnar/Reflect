#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_o_std104 \
#--teacher_exp_name=gc_o_tchr104 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_o_std101 \
#--teacher_exp_name=gc_o_tchr101 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &
#
#
##LSTM to Transformer
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_o_std102 \
#--teacher_exp_name=gc_o_tchr102 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &
#
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_o_std103 \
#--teacher_exp_name=gc_o_tchr103 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &
#
#
#Transformer to LSTM
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std204 \
--teacher_exp_name=gc_o_tchr204 \
--teacher_config=very_big_gpt_v10 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run1 &


CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std205 \
--teacher_exp_name=gc_o_tchr205 \
--teacher_config=very_big_gpt_v10 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run2 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std206 \
--teacher_exp_name=gc_o_tchr206 \
--teacher_config=very_big_gpt_v10 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run3 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std207 \
--teacher_exp_name=gc_o_tchr207 \
--teacher_config=very_big_gpt_v10 \
--student_config=biglstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run4 &


wait
