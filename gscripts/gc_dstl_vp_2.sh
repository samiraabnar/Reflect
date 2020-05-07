#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist


CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_lstm \
--student_exp_name=gc_f_std4101 \
--teacher_exp_name=gc_o_tch4101 \
--teacher_config=small_ugpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp4 > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_lstm \
--student_exp_name=gc_f_std4102 \
--teacher_exp_name=gc_o_tchr4102 \
--teacher_config=small_ugpt_v9 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp4 > run0 &

#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std4103 \
#--teacher_exp_name=gc_o_tchr4113 \
#--teacher_config=small_ugpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_lstm \
#--student_exp_name=gc_f_std4104 \
#--teacher_exp_name=gc_o_tchr4112 \
#--teacher_config=small_ugpt_v9 \
#--student_config=small_lstm_v4 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &
#
#
#CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std4133 \
#--teacher_exp_name=gc_o_tchr4123 \
#--teacher_config=small_ugpt_v9 \
#--student_config=small_ugpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &
#
#CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_f_std4130 \
#--teacher_exp_name=gc_o_tchr4130 \
#--teacher_config=small_ugpt_v9 \
#--student_config=small_ugpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &
#
#
#CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std4110 \
#--teacher_exp_name=gc_o_tchr4110 \
#--teacher_config=small_ugpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_gpt2_shared \
#--student_model=cl_bert \
#--student_exp_name=gc_f_std4111 \
#--teacher_exp_name=gc_o_tchr4111 \
#--teacher_config=small_ugpt_v9 \
#--student_config=small_gpt_v9 \
#--distill_mode=offline \
#--distill_config=pure_dstl_4_exp_vp4 > run0 &

CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std4120 \
--teacher_exp_name=gc_o_tchr4120 \
--teacher_config=small_ugpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp4 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_gpt2_shared \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std4121 \
--teacher_exp_name=gc_o_tchr4121 \
--teacher_config=small_ugpt_v9 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp4 > run0 &


wait