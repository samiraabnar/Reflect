#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist

#
CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_of_std5001 \
--teacher_exp_name=gc_o_tchr5011 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &

CUDA_VISIBLE_DEVICES=1 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_f_std5002 \
--teacher_exp_name=gc_o_tchr5020 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &

CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_f_std5003 \
--teacher_exp_name=gc_o_tchr5030 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &


CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=gc_f_std5004 \
--teacher_exp_name=gc_o_tchr5021 \
--teacher_config=small_lstm_v4 \
--student_config=small_lstm_v4 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &

############################################################


#CUDA_VISIBLE_DEVICES=2 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_o_std5030 \
#--teacher_exp_name=gc_o_tchr5030 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_ugpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp5 > run0 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/distill_main.py  \
#--task=word_sv_agreement_vp \
#--teacher_model=cl_lstm \
#--student_model=cl_gpt2_shared \
#--student_exp_name=gc_o_std5031 \
#--teacher_exp_name=gc_o_tchr5031 \
#--teacher_config=small_lstm_v4 \
#--student_config=small_ugpt_v9 \
#--distill_mode=online \
#--distill_config=pure_dstl_4_exp_vp5 > run0 &
#
#
CUDA_VISIBLE_DEVICES=4 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_f_std5010 \
--teacher_exp_name=gc_o_tchr5010 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &

CUDA_VISIBLE_DEVICES=5 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_bert \
--student_exp_name=gc_f_std5011 \
--teacher_exp_name=gc_o_tchr5011 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &
#
CUDA_VISIBLE_DEVICES=6 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std5020 \
--teacher_exp_name=gc_o_tchr5020 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &

CUDA_VISIBLE_DEVICES=7 python distill/distill_main.py  \
--task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_gpt2 \
--student_exp_name=gc_f_std5021 \
--teacher_exp_name=gc_o_tchr5021 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=offline \
--distill_config=pure_dstl_4_exp_vp5 > run0 &


wait