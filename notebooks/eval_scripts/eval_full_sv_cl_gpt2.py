import os
import tensorflow as tf
from util import constants
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.trainer import Trainer
from absl import app
from absl import flags
import numpy as np
from util.models import MODELS
from util.tasks import TASKS
from notebook_utils import *
import pandas as pd
import seaborn as sns; sns.set()
from collections import Counter

from tqdm import tqdm


log_dir = "../logs"
chkpt_dir = "../tf_ckpts"

task = TASKS['word_sv_agreement_vp'](task_params=get_task_params(),data_dir='../data')
cl_token = task.databuilder.sentence_encoder().encode(constants.bos)


modelz = {}
ckptz = {}




config={'student_exp_name':'gc_f_std144',
    'teacher_exp_name':'gc_o_tchr144',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_gpt2_shared',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_crs_slw',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# std_hparams=get_model_params(task, config['student_model'], config['student_config'])
# model, ckpt = get_student_model(config, task, std_hparams, cl_token)

# modelz['g2l_std144'] = model
# ckptz['g2l_std144'] = ckpt

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, teacger_ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

modelz['ug2l_tchr144'] = teacher_model
ckptz['ug2l_tchr144'] = teacger_ckpt



config={'student_exp_name':'gc_f_std145',
    'teacher_exp_name':'gc_o_tchr145',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_gpt2_shared',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_crs_slw',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# std_hparams=get_model_params(task, config['student_model'], config['student_config'])
# model, ckpt = get_student_model(config, task, std_hparams, cl_token)

# modelz['g2l_std145'] = model
# ckptz['g2l_std145'] = ckpt

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, teacger_ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

modelz['ug2l_tchr145'] = teacher_model
ckptz['ug2l_tchr145'] = teacger_ckpt



config={'student_exp_name':'gc_f_std146',
    'teacher_exp_name':'gc_o_tchr146',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_gpt2_shared',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_crs_slw',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# std_hparams=get_model_params(task, config['student_model'], config['student_config'])
# model, ckpt = get_student_model(config, task, std_hparams, cl_token)

# modelz['g2l_std146'] = model
# ckptz['g2l_std146'] = ckpt

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, teacger_ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

modelz['ug2l_tchr146'] = teacher_model
ckptz['ug2l_tchr146'] = teacger_ckpt



config={'student_exp_name':'gc_f_std147',
    'teacher_exp_name':'gc_o_tchr147',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_gpt2_shared',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_crs_slw',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# std_hparams=get_model_params(task, config['student_model'], config['student_config'])
# model, ckpt = get_student_model(config, task, std_hparams, cl_token)

# modelz['g2l_std147'] = model
# ckptz['g2l_std147'] = ckpt



tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, teacger_ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

modelz['ug2l_tchr147'] = teacher_model
ckptz['ug2l_tchr147'] = teacger_ckpt




infl_eng = inflect.engine()
verb_infl, noun_infl = gen_inflect_from_vocab(infl_eng, 'wiki.vocab')

keys = modelz.keys()

for key in keys:
    model = modelz[key]
    print('##################################')
    print(key, ckptz[key])
    distance_hits, distance_total, diff_hits, diff_total = evaluate_vp_cl(model, verb_infl, noun_infl, task)
    compute_and_print_acc_stats(distance_hits, distance_total, diff_hits, diff_total)