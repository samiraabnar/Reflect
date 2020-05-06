#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std1300 \
#--teacher_exp_name=gc_o_tchr1300 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp1 > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std2300 \
#--teacher_exp_name=gc_o_tchr2300 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp2 > run0 &

#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std7300 \
#--teacher_exp_name=gc_o_tchr7300 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp7 > run0 &

CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_lstm \
--student_exp_name=gc_o_std9300 \
--teacher_exp_name=gc_o_tchr9300 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp9 > run0 &

#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std1310 \
#--teacher_exp_name=gc_o_tchr1310 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp1 > run0 &
#
CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std9310 \
--teacher_exp_name=gc_o_tchr9310 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp9 > run0 &

#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std7310 \
#--teacher_exp_name=gc_o_tchr7310 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp7 > run0 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std8310 \
#--teacher_exp_name=gc_o_tchr8310 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp8 > run0 &




wait