#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std1000 \
#--teacher_exp_name=gc_o_tchr1000 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp1 > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std2000 \
#--teacher_exp_name=gc_o_tchr2000 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp2 > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std5000 \
--teacher_exp_name=gc_o_tchr5000 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp5 > run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_o_std6000 \
--teacher_exp_name=gc_o_tchr6000 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp6 > run0 &

#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std1010 \
#--teacher_exp_name=gc_o_tchr1010 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp1 > run0 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std2010 \
#--teacher_exp_name=gc_o_tchr2010 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp2 > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std5010 \
--teacher_exp_name=gc_o_tchr5010 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp5 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_o_std6010 \
--teacher_exp_name=gc_o_tchr6010 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp6 > run0 &




wait