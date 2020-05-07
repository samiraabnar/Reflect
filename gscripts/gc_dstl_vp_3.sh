#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_lstm \
--student_exp_name=gc_f_std9301 \
--teacher_exp_name=gc_o_tchr9301 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp9 > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_lstm \
--student_exp_name=gc_f_std9302 \
--teacher_exp_name=gc_o_tchr9302 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp9 > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_lstm \
--student_exp_name=gc_f_std9303 \
--teacher_exp_name=gc_o_tchr8323 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp9 > run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_lstm \
--student_exp_name=gc_f_std9304 \
--teacher_exp_name=gc_o_tchr8324 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp9 > run0 &

#
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_gpt2_shared \
--student_exp_name=gc_f_std8331 \
--teacher_exp_name=gc_o_tchr8321 \
--teacher_config=small_gpt_v9 \
--student_config=small_ugpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp8 > run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_gpt2_shared \
--student_exp_name=gc_f_std8332 \
--teacher_exp_name=gc_o_tchr8322 \
--teacher_config=small_gpt_v9 \
--student_config=small_ugpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp8 > run0 &

#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std8333 \
#--teacher_exp_name=gc_o_tchr8323 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_ugpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std8334 \
#--teacher_exp_name=gc_o_tchr8324 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_ugpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &


#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std8323 \
#--teacher_exp_name=gc_o_tchr8323 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std8322 \
#--teacher_exp_name=gc_o_tchr8322 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_f_std8321 \
#--teacher_exp_name=gc_o_tchr8321 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &

#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std8311 \
#--teacher_exp_name=gc_o_tchr8311 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std8312 \
#--teacher_exp_name=gc_o_tchr8322 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std8313 \
#--teacher_exp_name=gc_o_tchr8323 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_bert \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std8314 \
#--teacher_exp_name=gc_o_tchr8324 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &





wait