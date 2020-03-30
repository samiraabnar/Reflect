#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr100 \
--student_exp_name=gc_std100 \
--teacher_config=small_lstm_v4 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp100 &

CUDA_VISIBLE_DEVICES=1 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr101 \
--student_exp_name=gc_std101 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_3 > run_sv_lmvp101 &


CUDA_VISIBLE_DEVICES=2 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr102 \
--student_exp_name=gc_std102 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_00199_crs_slwfst_2 > run_sv_lmvp102 &

CUDA_VISIBLE_DEVICES=3 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_lstm_shared_emb \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr103 \
--student_exp_name=gc_std103 \
--teacher_config=lstm_drop31_v2 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_00199_crs_slwfst_3 > run_sv_lmvp103 &

CUDA_VISIBLE_DEVICES=4 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr104 \
--student_exp_name=gc_std104 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_00199_crs_slwfst_2 > run_sv_lmvp104 &

CUDA_VISIBLE_DEVICES=5 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr105 \
--student_exp_name=gc_std105 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_off_00199_crs_slwfst_3 > run_sv_lmvp105 &

CUDA_VISIBLE_DEVICES=6 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr106 \
--student_exp_name=gc_std106 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_2 > run_sv_lmvp106 &

CUDA_VISIBLE_DEVICES=7 python distill/repshare_main.py \
--teacher_task=word_sv_agreement_lm \
--student_task=word_sv_agreement_vp  \
--teacher_model=lm_gpt2_shared \
--student_model=cl_bert_shared \
--teacher_exp_name=gc_mor_dtchr107 \
--student_exp_name=gc_std107 \
--teacher_config=big_gpt_v5 \
--student_config=big_gpt_v5 \
--batch_size=64 \
--distill_mode=mrep_online \
--distill_config=rpdst_00199_crs_slwfst_3 > run_sv_lmvp107 &

wait