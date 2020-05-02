#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std104 \
#--teacher_exp_name=gc_o_tchr104 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std109 \
#--teacher_exp_name=gc_o_tchr109 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run1 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std110 \
#--teacher_exp_name=gc_o_tchr110 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run2 &
#
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std111 \
#--teacher_exp_name=gc_o_tchr111 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run3 &

#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std140 \
#--teacher_exp_name=gc_o_tchr140 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std136 \
#--teacher_exp_name=gc_o_tchr136 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run5 &
#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std157 \
#--teacher_exp_name=gc_o_tchr157 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run5 &
##
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std158 \
#--teacher_exp_name=gc_o_tchr158 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run6 &
#
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std159 \
#--teacher_exp_name=gc_o_tchr159 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run7 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std160 \
#--teacher_exp_name=gc_o_tchr160 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run8 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std161 \
#--teacher_exp_name=gc_o_tchr161 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run7 &

#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_o_std162 \
#--teacher_exp_name=gc_o_tchr162 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw > run7 &
#
CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std163 \
--teacher_exp_name=gc_o_tchr163 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run7 &
#
#
CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std164 \
--teacher_exp_name=gc_o_tchr164 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run7 &
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std153 \
#--teacher_exp_name=gc_o_tchr153 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run8 &

#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std154 \
#--teacher_exp_name=gc_o_tchr154 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run8 &

#
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std \
#--teacher_exp_name=gc_o_tchr \
#--teacher_config=small_lstm_v4 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run3 &


CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_gpt2_shared \
--student_exp_name=gc_f_std151 \
--teacher_exp_name=gc_o_tchr151 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run4 &
#
#
CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std165 \
--teacher_exp_name=gc_o_tchr165 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run4 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std166 \
#--teacher_exp_name=gc_o_tchr166 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std167 \
#--teacher_exp_name=gc_o_tchr167 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &
#
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std168 \
--teacher_exp_name=gc_o_tchr168 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run4 &
#
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std169 \
#--teacher_exp_name=gc_o_tchr169 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std170 \
#--teacher_exp_name=gc_o_tchr170 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std171 \
#--teacher_exp_name=gc_o_tchr171 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std173 \
#--teacher_exp_name=gc_o_tchr173 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw > run7 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std174 \
#--teacher_exp_name=gc_o_tchr174 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw > run7 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std175 \
#--teacher_exp_name=gc_o_tchr175 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw > run7 &
#
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std176 \
#--teacher_exp_name=gc_o_tchr176 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw > run7 &

wait