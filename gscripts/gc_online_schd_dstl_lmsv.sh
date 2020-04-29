#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist
# LSTM 2 LSTM

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std1 \
--teacher_exp_name=gc_tchr1 \
--teacher_config=lstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std2 \
--teacher_exp_name=gc_tchr2 \
--teacher_config=lstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run1 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std3 \
--teacher_exp_name=gc_tchr3 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std4 \
--teacher_exp_name=gc_tchr4 \
--teacher_config=biglstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run1 &


CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std5 \
--teacher_exp_name=gc_tchr5 \
--teacher_config=biglstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std6 \
--teacher_exp_name=gc_tchr6 \
--teacher_config=biglstm_drop31_v2 \
--student_config=lstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run1 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std7 \
--teacher_exp_name=gc_tchr7 \
--teacher_config=lstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl1_dstl_4_crs_fst3 > os_run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=lm_lstm_shared_emb \
--student_model=lm_lstm_shared_emb \
--student_exp_name=gc_std8 \
--teacher_exp_name=gc_tchr8 \
--teacher_config=lstm_drop31_v2 \
--student_config=biglstm_drop31_v2 \
--distill_mode=offline \
--distill_config=schdl2_dstl_4_crs_fst3 > os_run1 &


wait
