#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std108 \
--teacher_exp_name=gc_o_tchr108 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std109 \
--teacher_exp_name=gc_o_tchr109 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run1 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std110 \
--teacher_exp_name=gc_o_tchr110 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run2 &


CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std111 \
--teacher_exp_name=gc_o_tchr111 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run3 &


CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_gpt2 \
--student_exp_name=gc_o_std112 \
--teacher_exp_name=gc_o_tchr112 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl5_4_crs_slw > run4 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_gpt2 \
--student_exp_name=gc_o_std113 \
--teacher_exp_name=gc_o_tchr113 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run5 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_gpt2 \
--student_exp_name=gc_o_std114 \
--teacher_exp_name=gc_o_tchr114 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run6 &


CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_bert \
--student_model=cl_gpt2 \
--student_exp_name=gc_o_std115 \
--teacher_exp_name=gc_o_tchr115 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run7 &