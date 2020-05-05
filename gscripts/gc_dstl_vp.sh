#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std2000 \
--teacher_exp_name=gc_o_tchr2000 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &


CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std2010 \
--teacher_exp_name=gc_o_tchr2010 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &


CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_fo_std2020 \
--teacher_exp_name=gc_o_tchr2020 \
--teacher_config=small_gpt_v4 \
--student_config=small_gpt_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_lstm \
--student_exp_name=gc_o_std2030 \
--teacher_exp_name=gc_o_tchr2030 \
--teacher_config=small_gpt_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &


CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_bert \
--student_exp_name=gc_o_std2040 \
--teacher_exp_name=gc_o_tchr2040 \
--teacher_config=small_gpt_v4 \
--student_config=small_gpt_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_lstm \
--student_exp_name=gc_o_std2050 \
--teacher_exp_name=gc_o_tchr2050 \
--teacher_config=small_gpt_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &


CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_o_std2060 \
--teacher_exp_name=gc_o_tchr2060 \
--teacher_config=small_gpt_v4 \
--student_config=small_gpt_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_lstm \
--student_exp_name=gc_o_std2070 \
--teacher_exp_name=gc_o_tchr2070 \
--teacher_config=small_gpt_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_hld3 > run0 &

wait