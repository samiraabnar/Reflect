#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std1 \
--teacher_exp_name=gc_o_tchr1 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl5_4_crs_slw > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std2 \
--teacher_exp_name=gc_o_tchr2 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std3 \
--teacher_exp_name=gc_o_tchr3 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst > run0 &


CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert \
--student_model=cl_bert \
--student_exp_name=gc_o_std4 \
--teacher_exp_name=gc_o_tchr4 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl5_4_crs_fst > run0 &


CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert_shared \
--student_model=cl_bert_shared \
--student_exp_name=gc_o_std5 \
--teacher_exp_name=gc_o_tchr5 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl5_4_crs_slw > run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert_shared \
--student_model=cl_bert_shared \
--student_exp_name=gc_o_std6 \
--teacher_exp_name=gc_o_tchr6 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_slw > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert_shared \
--student_model=cl_bert_shared \
--student_exp_name=gc_o_std7 \
--teacher_exp_name=gc_o_tchr7 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl_4_crs_fst > run0 &


CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  --task=word_sv_agreement_lm \
--teacher_model=cl_bert_shared \
--student_model=cl_bert_shared \
--student_exp_name=gc_o_std8 \
--teacher_exp_name=gc_o_tchr8 \
--teacher_config=small_gpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=pure_dstl5_4_crs_fst > run0 &