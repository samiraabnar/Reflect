#!/usr/bin/env bash

#!/usr/bin/env bash

conda activate indist

cd ~/Codes/InDist

export PYTHONPATH=$PYTHONPATH:/home/dehghani/Codes/InDist

CUDA_VISIBLE_DEVICES=0 python distill/distill_main.py  --task=word_sv_agreement_vp \
--teacher_model=cl_lstm \
--student_model=cl_lstm \
--student_exp_name=af_std1 \
--teacher_exp_name=af_tchr1 \
--teacher_config=small_lstm_v4 \
--student_config=small_gpt_v9 \
--distill_mode=online \
--distill_config=dstl5_910_crs_slwfst_2 > off_vp_run0 &