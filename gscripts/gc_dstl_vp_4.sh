#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


#CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std9200 \
#--teacher_exp_name=gc_o_tchr9200 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp9 > run0 &
#
#CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_lstm \
#--student_exp_name=gc_o_std9201 \
#--teacher_exp_name=gc_o_std9202 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp9 > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_lstm \
--student_exp_name=gc_f_std9202 \
--teacher_exp_name=gc_o_tchr8222 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp9 > run0 &

CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_lstm \
--student_exp_name=gc_f_std9203 \
--teacher_exp_name=gc_o_st8223 \
--teacher_config=small_gpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp9 > run0 &

#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std8231 \
#--teacher_exp_name=gc_o_tchr8231 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_ugpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std8232 \
#--teacher_exp_name=gc_o_tchr8232 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_ugpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_o_std8223 \
#--teacher_exp_name=gc_o_tchr823 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
##
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2 \
#--student_model=cl_gpt2 \
#--student_exp_name=gc_o_std8224 \
#--teacher_exp_name=gc_o_tchr8224 \
#--teacher_config=small_gpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp8 > run0 &
#
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_f_std8211 \
--teacher_exp_name=gc_o_tchr8211 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp8 > run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_f_std8212 \
--teacher_exp_name=gc_o_tchr8222 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp8 > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_of_std8213 \
--teacher_exp_name=gc_o_tchr8213 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp8 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2 \
--student_model=cl_bert \
--student_exp_name=gc_f_std8213 \
--teacher_exp_name=gc_o_tchr8224 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp8 > run0 &





wait