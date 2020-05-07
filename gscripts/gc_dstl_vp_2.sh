#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist

#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std3100 \
#--teacher_exp_name=gc_o_tch3100 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_slw_vp3 > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std6100 \
#--teacher_exp_name=gc_o_tchr6100 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_slw_vp6 > run0 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_o_std3130 \
#--teacher_exp_name=gc_o_tchr3130 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp3 > run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std4130 \
#--teacher_exp_name=gc_o_tchr4130 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &
#
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std3110 \
#--teacher_exp_name=gc_o_tchr3110 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp3 > run0 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std4110 \
#--teacher_exp_name=gc_o_tchr4110 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_o_tchr3120 \
--teacher_exp_name=gc_o_tchr3120 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_vp3 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_bert \
--student_exp_name=gc_o_tchr4120 \
--teacher_exp_name=gc_o_tchr4120 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_exp_vp4 > run0 &


wait