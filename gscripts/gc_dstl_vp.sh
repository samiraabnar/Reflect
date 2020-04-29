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
#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std112 \
#--teacher_exp_name=gc_o_tchr112 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run4 &

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2_shared \
--student_exp_name=gc_f_std136 \
--teacher_exp_name=gc_o_tchr136 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run5 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std140 \
--teacher_exp_name=gc_o_tchr140 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run5 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std141 \
--teacher_exp_name=gc_o_tchr141 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run6 &


CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std142 \
--teacher_exp_name=gc_o_tchr142 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run7 &

CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std143 \
--teacher_exp_name=gc_o_tchr143 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_crs_slw > run8 &
#
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std131 \
#--teacher_exp_name=gc_o_tchr131 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run3 &


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std125 \
#--teacher_exp_name=gc_o_tchr125 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_crs_slw > run3 &


###
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std131 \
#--teacher_exp_name=gc_o_tchr131 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_crs_slw > run4 &

wait