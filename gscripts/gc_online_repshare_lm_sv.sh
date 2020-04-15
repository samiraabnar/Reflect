#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr120 \
--student_exp_name=gc_std120 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_00199_crs_slw_550 > run_sv_lm1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr121 \
--student_exp_name=gc_std121 \
--teacher_config=big_gpt_v5 \
--student_config=very_big_gpt_v10 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_00199_crs_slw_550 > run_sv_lm2 &
#
#CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
#--task=word_sv_agreement_lm  \
#--teacher_model=lm_gpt2 \
#--student_model=lm_gpt2 \
#--teacher_exp_name=gc_or_dtchr102 \
#--student_exp_name=gc_std102 \
#--teacher_config=big_gpt_v5 \
#--student_config=big_gpt_v5 \
#--batch_size=64 \
#--distill_mode=rep_online \
#--distill_config=rpdst_019_crs_slwfst_52 > run_sv_lm3 &
#
#CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
#--task=word_sv_agreement_lm  \
#--teacher_model=lm_gpt2 \
#--student_model=lm_gpt2 \
#--teacher_exp_name=gc_or_dtchr103 \
#--student_exp_name=gc_std103 \
#--teacher_config=big_gpt_v5 \
#--student_config=big_gpt_v5 \
#--batch_size=64 \
#--distill_mode=rep_online \
#--distill_config=rpdst_019_crs_slwfst_53 > run_sv_lm4 &


#CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
#--task=word_sv_agreement_lm  \
#--teacher_model=lm_gpt2 \
#--student_model=lm_gpt2 \
#--teacher_exp_name=gc_or_dtchr104 \
#--student_exp_name=gc_std104 \
#--teacher_config=very_big_gpt_v10 \
#--student_config=big_gpt_v5 \
#--batch_size=64 \
#--distill_mode=rep_online \
#--distill_config=rpdst_019_crs_slwfst_81 > run_sv_lm5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr125 \
--student_exp_name=gc_std125 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_00199_crs_fst_550 > run_sv_lm6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr126 \
--student_exp_name=gc_std126 \
--teacher_config=big_gpt_v5 \
--student_config=very_big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_00199_crs_fst_550 > run_sv_lm7 &
#
#CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
#--task=word_sv_agreement_lm  \
#--teacher_model=lm_gpt2 \
#--student_model=lm_gpt2_shared \
#--teacher_exp_name=gc_or_dtchr107 \
#--student_exp_name=gc_std107 \
#--teacher_config=big_gpt_v5 \
#--student_config=big_gpt_v5 \
#--batch_size=64 \
#--distill_mode=rep_online \
#--distill_config=rpdst_019_crs_slwfst_52 > run_sv_lm8 &


wait