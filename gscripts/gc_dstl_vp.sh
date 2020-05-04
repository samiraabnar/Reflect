#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std204 \
#--teacher_exp_name=gc_o_tchr104 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std205 \
#--teacher_exp_name=gc_o_tchr105 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run0 &
#

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std206 \
#--teacher_exp_name=gc_o_tchr106 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run0 &
#

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std207 \
#--teacher_exp_name=gc_o_tchr107 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run0 &
#
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std209 \
#--teacher_exp_name=gc_o_tchr109 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run1 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std210 \
#--teacher_exp_name=gc_o_tchr110 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run2 &
#
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std211 \
#--teacher_exp_name=gc_o_tchr111 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run3 &

#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std240 \
#--teacher_exp_name=gc_o_tchr140 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run4 &

#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std236 \
#--teacher_exp_name=gc_o_tchr136 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run5 &
#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std257 \
#--teacher_exp_name=gc_o_tchr157 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run5 &
##
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std258 \
#--teacher_exp_name=gc_o_tchr158 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run6 &
#
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std259 \
#--teacher_exp_name=gc_o_tchr159 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run7 &
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

#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_o_std162 \
#--teacher_exp_name=gc_o_tchr162 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run7 &
#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std163 \
#--teacher_exp_name=gc_o_tchr163 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run7 &
##
##
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std164 \
#--teacher_exp_name=gc_o_tchr164 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run7 &
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


#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std251 \
#--teacher_exp_name=gc_o_tchr151 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run4 &
###
##
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std165 \
#--teacher_exp_name=gc_o_tchr165 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &
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
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std168 \
#--teacher_exp_name=gc_o_tchr168 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &

#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std269 \
#--teacher_exp_name=gc_o_tchr169 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run4 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std270 \
#--teacher_exp_name=gc_o_tchr170 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run4 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std271 \
#--teacher_exp_name=gc_o_tchr171 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run4 &
#
CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_f_std172 \
--teacher_exp_name=gc_o_tchr172 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run4 &


#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std273 \
#--teacher_exp_name=gc_o_tchr173 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run7 &
##
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std274 \
#--teacher_exp_name=gc_o_tchr174 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run7 &
##
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std275 \
#--teacher_exp_name=gc_o_tchr175 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run7 &
#
#
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std176 \
#--teacher_exp_name=gc_o_tchr176 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run7 &


#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std276 \
#--teacher_exp_name=gc_o_tchr276 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl5_4_crs_slw > run7 &

#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std177 \
#--teacher_exp_name=gc_o_tchr177 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl5_4_crs_slw > run7 &
#
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_bert \
#--student_exp_name=gc_o_std178 \
#--teacher_exp_name=gc_o_tchr18 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl5_4_crs_slw > run7 &

wait