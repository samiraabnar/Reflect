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
import logging
tf.get_logger().setLevel(logging.ERROR)

log_dir = "../logs"
chkpt_dir = "../tf_ckpts"

task = TASKS['word_sv_agreement_vp'](task_params=get_task_params(batch_size=512),data_dir='../data')
cl_token = task.databuilder.sentence_encoder().encode(constants.bos)


models = []
labels = []


#Bert to LSTM

config={'student_exp_name':'gc_f_std9303',
    'teacher_exp_name':'gc_o_tchr8323',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_bert',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_exp_vp9',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# config['distill_mode'] = 'online'
# config['student_exp_name'] = config['student_exp_name'].replace('_f_', '_o_')

std_hparams=get_model_params(task, config['student_model'], config['student_config'])
std_hparams.output_attentions = True
std_hparams.output_embeddings = True

student_model, ckpt = get_student_model(config, task, std_hparams, cl_token)

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

models.append(teacher_model)
labels.append('bert2lstm_1')

config={'student_exp_name':'gc_f_std9304',
    'teacher_exp_name':'gc_o_tchr8324',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_bert',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_exp_vp9',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# config['distill_mode'] = 'online'
# config['student_exp_name'] = config['student_exp_name'].replace('_f_', '_o_')

std_hparams=get_model_params(task, config['student_model'], config['student_config'])
std_hparams.output_attentions = True
std_hparams.output_embeddings = True

student_model, ckpt = get_student_model(config, task, std_hparams, cl_token)

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

models.append(teacher_model)
labels.append('bert2lstm_2')

config={'student_exp_name':'gc_f_std9301',
    'teacher_exp_name':'gc_o_tchr9301',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_bert',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_exp_vp9',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }


# config['distill_mode'] = 'online'
# config['student_exp_name'] = config['student_exp_name'].replace('_f_', '_o_')

std_hparams=get_model_params(task, config['student_model'], config['student_config'])
std_hparams.output_attentions = True
std_hparams.output_embeddings = True

student_model, ckpt = get_student_model(config, task, std_hparams, cl_token)

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

models.append(teacher_model)
labels.append('bert2lstm_3')

config={'student_exp_name':'gc_f_std9302',
    'teacher_exp_name':'gc_o_tchr9302',
    'task_name':'word_sv_agreement_vp',
    'teacher_model':'cl_bert',
    'student_model':'cl_lstm',
    'teacher_config':'small_gpt_v9',
    'student_config':'small_lstm_v4',
    'distill_config':'pure_dstl_4_exp_vp9',
    'distill_mode':'offline',
    'chkpt_dir':'../tf_ckpts',
     }

# config['distill_mode'] = 'online'
# config['student_exp_name'] = config['student_exp_name'].replace('_f_', '_o_')

std_hparams=get_model_params(task, config['student_model'], config['student_config'])
std_hparams.output_attentions = True
std_hparams.output_embeddings = True

student_model, ckpt = get_student_model(config, task, std_hparams, cl_token)

tchr_hparams=get_model_params(task, config['teacher_model'], config['teacher_config'])
teacher_model, ckpt = get_teacher_model(config, task, tchr_hparams, cl_token)

models.append(teacher_model)
labels.append('bert2lstm_4')




keys = labels



import tensorflow_probability as tfp

def test_for_calibration(model, task, n_bins=10):
    preds = []
    correct_class_probs = []
    predicted_class_probs = []
    pred_logits = []
    y_trues = []
    batch_count = task.n_valid_batches
    for x, y in task.valid_dataset:
        y = tf.cast(y, tf.int32)
        logits = model(x)
        pred_logits.extend(logits.numpy())
        pred = tf.argmax(logits, axis=-1)
        prob = task.get_probs_fn()(logits, labels=y, temperature=1)
        preds.extend(pred.numpy())
        y_trues.extend(y.numpy())
        batch_indexes = tf.cast(tf.range(len(y), dtype=tf.int32), dtype=tf.int32)
        true_indexes = tf.concat([batch_indexes[:,None], y[:,None]], axis=1)
        pred_indexes = tf.concat([batch_indexes[:,None], tf.cast(pred[:,None], tf.int32)], axis=1)

        correct_class_probs.extend(tf.gather_nd(prob, true_indexes).numpy())
        predicted_class_probs.extend(tf.gather_nd(prob, pred_indexes).numpy())

        batch_count -= 1
        if batch_count == 0:
            break

    model_accuracy = np.asarray(preds) == np.asarray(y_trues)

    return model_accuracy, predicted_class_probs, correct_class_probs, pred_logits, y_trues


# for key in keys:
#     model = models[key]
#     print('##################################')
#     train = model.evaluate(task.train_dataset, steps=task.n_train_batches)
#     valid = model.evaluate(task.valid_dataset, steps=task.n_valid_batches)
#     test = model.evaluate(task.test_dataset, steps=task.n_test_batches)
#     print(key)
#     print(train[0],'\t',train[1],'\t',train[2],'\t', valid[0],'\t', valid[1],'\t', valid[2], '\t', test[0], '\t', test[1], '\t', test[2])
    
# for key in keys:
#     model = models[key]
#     print('##################################')

#     model_accuracy, predicted_class_probs, correct_class_probs, model_logits, model_trues= test_for_calibration(model, task, n_bins=20)
#     model_ece = tfp.stats.expected_calibration_error(
#         1000000,
#         logits=model_logits,
#         labels_true=model_trues,
#     )
#     print(model_ece.numpy())
        
    
infl_eng = inflect.engine()
verb_infl, noun_infl = gen_inflect_from_vocab(infl_eng, 'wiki.vocab')


print(labels)
for key,model in zip(labels,models):
    print('##################################')
    print(key)
    distance_hits, distance_total, diff_hits, diff_total = evaluate_vp_cl(model, verb_infl, noun_infl, task)
    compute_and_print_acc_stats(distance_hits, distance_total, diff_hits, diff_total)
    
    