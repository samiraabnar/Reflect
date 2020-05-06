#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std1200 \
#--teacher_exp_name=gc_o_tchr1200 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp1 > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std2200 \
#--teacher_exp_name=gc_o_tchr2200 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp2 > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_lstm \
--student_exp_name=gc_o_std7200 \
--teacher_exp_name=gc_o_tchr7200 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp7 > run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_lstm \
--student_exp_name=gc_o_std8200 \
--teacher_exp_name=gc_o_tchr8200 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp8 > run0 &

#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std1210 \
#--teacher_exp_name=gc_o_tchr1210 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp1 > run0 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std2210 \
#--teacher_exp_name=gc_o_tchr2210 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw_vp2 > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_o_std7210 \
--teacher_exp_name=gc_o_tchr7210 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp7 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_o_std8210 \
--teacher_exp_name=gc_o_tchr8210 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw_vp8 > run0 &




wait