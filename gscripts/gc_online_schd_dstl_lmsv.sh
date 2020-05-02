#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2 LSTM
#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std11 \
#--teacher_exp_name=gc_tchr11 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std13 \
#--teacher_exp_name=gc_tchr13 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &
#
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std211 \
#--teacher_exp_name=gc_tchr211 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std213 \
#--teacher_exp_name=gc_tchr213 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std11 \
#--teacher_exp_name=gc_tchr11 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &
#
CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_std301 \
--teacher_exp_name=gc_tchr301 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--distill_mode=online \
--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &


CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_std302 \
--teacher_exp_name=gc_tchr302 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--distill_mode=online \
--distill_config=schdl2_dstl_4_crs_fst4 > os_run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_std303 \
--teacher_exp_name=gc_tchr303 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--distill_mode=online \
--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_gpt2 \
--student_exp_name=gc_std304 \
--teacher_exp_name=gc_tchr304 \
--teacher_config=biglstm_drop31_v2 \
--student_config=big_gpt_v5 \
--distill_mode=online \
--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &

CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std200 \
--teacher_exp_name=gc_tchr200 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std201 \
--teacher_exp_name=gc_tchr201 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std202 \
--teacher_exp_name=gc_tchr202 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std203 \
--teacher_exp_name=gc_tchr203 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run0 &

#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std201 \
#--teacher_exp_name=gc_tchr130 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &


#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std14 \
#--teacher_exp_name=gc_tchr14 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=schdl1_dstl_4_crs_fst4 > os_run0 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std14 \
#--teacher_exp_name=gc_tchr14 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=schdl1_dstl_4_crs_fst4 > os_run0 &

#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std15 \
#--teacher_exp_name=gc_tchr15 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &


#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std10 \
#--teacher_exp_name=gc_tchr10 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=schdl1_dstl_4_crs_fst4 > os_run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_gpt2 \
#--student_exp_name=gc_std11 \
#--teacher_exp_name=gc_tchr11 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=very_big_gpt_v10 \
#--distill_mode=online \
#--distill_config=schdl2_dstl_4_crs_fst4 > os_run1 &




#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_std9 \
#--teacher_exp_name=gc_tchr9 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=online \
#--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_std10 \
#--teacher_exp_name=gc_tchr10 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=online \
#--distill_config=schdl1_dstl_4_crs_fst3 > os_run1 &


#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_std5 \
#--teacher_exp_name=gc_tchr5 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_std6 \
#--teacher_exp_name=gc_tchr6 \
#--teacher_config=biglstm_drop31_v2 \
#--student_config=lstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst3 > os_run1 &
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_std7 \
#--teacher_exp_name=gc_tchr7 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &

#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  --task=word_sv_agreement_lm \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=lm_lstm_shared_emb \
#--student_exp_name=gc_std8 \
#--teacher_exp_name=gc_tchr8 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=biglstm_drop31_v2 \
#--distill_mode=offline \
#--distill_config=schdl2_dstl_4_crs_fst3 > os_run1 &


wait
