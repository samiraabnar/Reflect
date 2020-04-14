#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr91 \
--student_exp_name=gc_std91 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm1 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2_shared \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr92 \
--student_exp_name=gc_std92 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm2 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr93 \
--student_exp_name=gc_std93 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm3 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2_shared \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr94 \
--student_exp_name=gc_std94 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm4 &


CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2 \
--teacher_exp_name=gc_or_dtchr95 \
--student_exp_name=gc_std95 \
--teacher_config=very_big_gpt_v10 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm5 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2_shared \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr96 \
--student_exp_name=gc_std96 \
--teacher_config=very_big_gpt_v10 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm6 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr97 \
--student_exp_name=gc_std97 \
--teacher_config=very_big_gpt_v10 \
--student_config=very_big_gpt_v10 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm7 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--task=word_sv_agreement_lm  \
--teacher_model=lm_gpt2 \
--student_model=lm_gpt2_shared \
--teacher_exp_name=gc_or_dtchr98 \
--student_exp_name=gc_std98 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=rep_online \
--distill_config=rpdst_019_crs_slwfst_55 > run_sv_lm8 &

wait