#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std6001 \
--teacher_exp_name=gc_o_tchr6001 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp6 > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std6002 \
--teacher_exp_name=gc_o_tchr6002 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp6 > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std9001 \
--teacher_exp_name=gc_o_tchr9001 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp9 > run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std9002 \
--teacher_exp_name=gc_o_tchr9002 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp9 > run0 &


CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std6011 \
--teacher_exp_name=gc_o_tchr6011 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp6 > run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std6012 \
--teacher_exp_name=gc_o_tchr6012 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp6 > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std9011 \
--teacher_exp_name=gc_o_tchr9011 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp9 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std9012 \
--teacher_exp_name=gc_o_tchr9012 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_slw_vp9 > run0 &


wait