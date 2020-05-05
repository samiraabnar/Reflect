#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2



#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_o_std125 \
#--teacher_exp_name=gc_o_tchr125 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_o_std126 \
#--teacher_exp_name=gc_o_tchr126 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_o_std127 \
#--teacher_exp_name=gc_o_tchr127 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_o_std128 \
#--teacher_exp_name=gc_o_tchr128 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
##
##
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_o_std129 \
#--teacher_exp_name=gc_o_tchr129 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &

#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_f_std117 \
#--teacher_exp_name=gc_o_tchr117 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=big_gpt_v5 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_f_std118 \
#--teacher_exp_name=gc_o_tchr118 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=big_gpt_v5 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
##

#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_f_std116 \
#--teacher_exp_name=gc_o_tchr116 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=big_gpt_v5 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_f_std105 \
#--teacher_exp_name=gc_o_tchr105 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_f_std106 \
#--teacher_exp_name=gc_o_tchr106 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_o_std107 \
#--teacher_exp_name=gc_o_tchr107 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst4 > o_run0 &
##
#
##
##
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std08 \
#--teacher_exp_name=gc_o_tchr108 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &

#
##LSTM to Transformer
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std109 \
#--teacher_exp_name=gc_o_tchr109 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &
#
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std110 \
#--teacher_exp_name=gc_o_tchr110 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &


#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std111 \
#--teacher_exp_name=gc_o_tchr111 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst3 > o_run0 &
#
#
#Transformer to LSTM
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_gpt2 \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std204 \
#--teacher_exp_name=gc_o_tchr204 \
#--teacher_config=very_big_gpt_v10 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst5 > o_run1 &

#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_gpt2 \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std205 \
#--teacher_exp_name=gc_o_tchr205 \
#--teacher_config=very_big_gpt_v10 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst5 > o_run2 &
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_gpt2 \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std206 \
#--teacher_exp_name=gc_o_tchr206 \
#--teacher_config=very_big_gpt_v10 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst5 > o_run3 &
##
#CUDA_VISIBLE_DEVICE=5 python distill/distill_main.py  \
#--task=word_sv_agreement_lm \
#--teacher_model=lm_gpt2 \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_f_std210 \
#--teacher_exp_name=gc_o_tchr210 \
#--teacher_config=big_gpt_v5 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_fst5 > o_run4 &
#
#
CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std211 \
--teacher_exp_name=gc_o_tchr211 \
--teacher_config=big_gpt_v5 \
--student_config=lstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run4 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_f_std212 \
--teacher_exp_name=gc_o_tchr212 \
--teacher_config=big_gpt_v5 \
--student_config=lstm_drop31_v2 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_fst5 > o_run4 &


CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_lm \
--teacher_model=lm_gpt2 \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_o_std213 \
--teacher_exp_name=gc_o_tchr213 \
--teacher_config=big_gpt_v5 \
--student_config=lstm_drop31_v2 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst5 > o_run4 &


wait
