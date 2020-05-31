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
from tqdm import tqdm

log_dir = "../logs"
chkpt_dir = "../tf_ckpts"

task = TASKS['word_sv_agreement_lm'](task_params=get_task_params(),data_dir='../data')
cl_token = task.databuilder.sentence_encoder().encode(constants.bos)


modelz = {}
ckptz = {}



config = {'model_name':'lm_lstm_shared_emb',
            'model_config':'lstm_drop31_v2',
            'learning_rate':0.001,
            'exp_name':'lisa_crs_fst_offlineteacher_v23',
            'chkpt_dir': '../tf_ckpts'
    }
hparams=get_model_params(task, config['model_name'], config['model_config'])
hparams.output_attentions = True
hparams.output_embeddings = True

lstm1, lstm_ckpt1 = get_model(config, task, hparams, cl_token)
modelz['lstm1'] = lstm1
ckptz['lstm1'] = lstm_ckpt1


config = {'model_name':'lm_lstm_shared_emb',
            'model_config':'lstm_drop31_v2',
            'learning_rate':0.001,
            'exp_name':'lisa_crs_fst_offlineteacher_v24',
            'chkpt_dir': '../tf_ckpts'
    }
hparams=get_model_params(task, config['model_name'], config['model_config'])
hparams.output_attentions = True
hparams.output_embeddings = True

lstm2, lstm_ckpt2 = get_model(config, task, hparams, cl_token)
modelz['lstm2'] = lstm2
ckptz['lstm2'] = lstm_ckpt2

config = {'model_name':'lm_lstm_shared_emb',
            'model_config':'lstm_drop31_v2',
            'learning_rate':0.001,
            'exp_name':'lisa_crs_fst_offlineteacher_v25',
            'chkpt_dir': '../tf_ckpts'
    }
hparams=get_model_params(task, config['model_name'], config['model_config'])
hparams.output_attentions = True
hparams.output_embeddings = True

lstm3, lstm_ckpt3 = get_model(config, task, hparams, cl_token)
modelz['lstm3'] = lstm3
ckptz['lstm3'] = lstm_ckpt3



keys = ['lstm1', 'lstm2']


print("Evaluations ...")
for key in keys:
    model = modelz[key]
    print('##################################')
    print(ckptz[key])
    
    train = model.evaluate(task.train_dataset, steps=task.n_train_batches)
    valid = model.evaluate(task.valid_dataset, steps=task.n_valid_batches)
    test = model.evaluate(task.test_dataset, steps=task.n_test_batches)

    print("train:", train)
    print("valid:", valid)
    print("test:", test)