#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr200 \
--student_exp_name=gc_std200 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_gpt_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp200 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr201 \
--student_exp_name=gc_std201 \
--teacher_config=very_big_gpt_v10 \
--student_config=small_gpt_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp201 &

CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr202 \
--student_exp_name=gc_std202 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_gpt_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_2 > run_sv_lmvp202 &


#CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
#--teacher_task=word_sv_agreement_lm \
#--student_task=word_sv_agreement_vp  \
#--teacher_model=lm_lstm_shared_emb \
#--student_model=cl_lstm \
#--teacher_exp_name=gc_mor_dtchr203 \
#--student_exp_name=gc_std203 \
#--teacher_config=lstm_drop31_v2 \
#--student_config=small_lstm_v4 \
#--batch_size=64 \
#--distill_mode=mrep_online \
#--distill_config=rpdst_00199_crs_slwfst_3 > run_sv_lmvp203 &

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr204 \
--student_exp_name=gc_std204 \
--teacher_config=very_big_gpt_v10 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_3 > run_sv_lmvp204 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr205 \
--student_exp_name=gc_std205 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_010_crs_slwfst_3 > run_sv_lmvp205 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr206 \
--student_exp_name=gc_std206 \
--teacher_config=lstm_drop31_v2 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp206 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2 \
--student_model=cl_lstm \
--teacher_exp_name=gc_mor_dtchr207 \
--student_exp_name=gc_std207 \
--teacher_config=very_big_gpt_v10 \
--student_config=small_lstm_v4 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp207 &


wait