''' Evaluate word based language models on the subject verb agreement task.

Codes adapted from:

Example Run:
python tasks/evaluations/lm_sv_agreement_eval.py \
--exp_name=tune_withl2_withpunc \
--model_name=lm_gpt2_shared \
--model_config=very_big_gpt_v10 \
--train_config=adam_slow
'''
import os

from tasks.sv_agreement import WordSvAgreementLM
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.trainer import Trainer
from util import constants
from collections import Counter
from tqdm import tqdm
from tf2_models.metrics import *
import numpy as np
from absl import flags
from absl import app
from util.models import MODELS
from util.text_util import gen_inflect_from_vocab


FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', 'logs', ' log dir path')
flags.DEFINE_string('chkpt_dir', 'chkpt_dir', ' chkpt_dir path')
flags.DEFINE_string('prefix', 'prefix', ' prefix')
flags.DEFINE_string('exp_name', 'tune_withl2_withpunc', 'tune_withl2_withpunc | withl2_batchsumloss_withpunc')
flags.DEFINE_string('model_config', 'very_big_gpt_v10', 'big_gpt_v5 | very_big_gpt_v10| lstm_drop31_v2')
flags.DEFINE_string('model_name', 'lm_gpt2_shared', 'lm_gpt2_shared | lm_gpt1 | lm_lstm_shared_emb')
flags.DEFINE_string('train_config', 'adam_slow', ' adam_slow | radam_fast')

hparams = flags.FLAGS

def compute_and_print_acc_stats(distance_hits, distance_total, diff_hits, diff_total):
  ''' Computes and prints accuracy based on hits

  :param distance_hits:
  :param distance_total:
  :param diff_hits:
  :param diff_total:
  :return: None
  '''
  dis_acc = {}
  dis_acc = np.zeros(17)
  dif_acc = np.zeros(5)
  total_nominator = 0.0
  total_denominator = 0.0
  print('Accuracy by distance')
  for k in sorted(distance_hits.keys()):
    v = distance_hits[k]
    acc = v / distance_total[k]
    dis_acc[k] = acc
    print("%d | %.2f" % (k, acc), distance_total[k])
    total_nominator += v
    total_denominator += distance_total[k]

  print('Accuracy by intervenings:')
  for k in sorted(diff_hits.keys()):
    v = diff_hits[k]
    acc = v * 1. / diff_total[k]
    print("%d | %.2f" % (k, acc), diff_total[k])
    dif_acc[k] = acc

  print("Total accuracy:", total_nominator/total_nominator)

def evaluate_vp(model, task):
  ''' Computes the accuracy statistics of the given model on the subject verb agreement task.

  :param model: the models to be evaluated
  :param task:
  :return: distance_hits, distance_total, diff_hits, diff_total
  '''

  verb_infl, noun_infl = gen_inflect_from_vocab('data/tal_agreement/wiki.vocab')

  distance_hits = Counter()
  distance_total = Counter()
  diff_hits = Counter()
  diff_total = Counter()

  test_data = task.databuilder.as_dataset(split='test', batch_size=1000)
  for example in tqdm(test_data):
    encoded_sentences = example['sentence']
    s_shape = tf.shape(encoded_sentences)
    batch_size, length = s_shape[0], s_shape[1]
    bos = tf.ones((batch_size, 1), dtype=tf.int64) * task.databuilder.sentence_encoder().encode(constants.bos)
    eos = tf.ones((batch_size, 1), dtype=tf.int64) * task.databuilder.sentence_encoder().encode(constants.eos)

    encoded_sentences = tf.concat([bos, encoded_sentences, eos], axis=1)

    actual_verbs = example['verb']
    inflected_verbs = [verb_infl[v.decode("utf-8")] for v in actual_verbs.numpy()]
    verb_indexes = example['verb_position']
    distances = example['distance'].numpy()
    nz = example['n_intervening'].numpy()
    n_diffs = example['n_diff_intervening'].numpy()

    actual_verb_indexes = [task.databuilder.sentence_encoder().encode(v)[0] for v in actual_verbs.numpy()]
    inflected_verb_indexes = [task.databuilder.sentence_encoder().encode(v)[0] for v in inflected_verbs]

    scores = model(encoded_sentences)
    actual_batch_indexes = [(i, verb_indexes[i], actual_verb_indexes[i]) for i in range(len(verb_indexes))]
    actual_scores = tf.compat.v2.gather_nd(scores, actual_batch_indexes)

    inflected_batch_indexes = [(i, verb_indexes[i], inflected_verb_indexes[i]) for i in range(len(verb_indexes))]
    infelected_scores = tf.compat.v2.gather_nd(scores, inflected_batch_indexes)

    corrects = actual_scores > infelected_scores
    for i, c in enumerate(corrects):
      if nz[i] > 4 or distances[i] > 16:
        continue

      distance_total[distances[i]] += 1
      distance_hits[distances[i]] += int(c)
      if nz[i] == n_diffs[i]:
        n = nz[i]
        diff_total[n] += 1
        diff_hits[n] += int(c)

  return  distance_hits, distance_total, diff_hits, diff_total


def main(argv):
  task = WordSvAgreementLM(task_params=get_task_params(), data_dir='data')

  # Create the Model
  model_params = get_model_params(task, hparams.model_name, hparams.model_config)
  print("model_params: ", model_params.__dict__)
  cl_token = task.databuilder.sentence_encoder().encode(constants.bos)
  model = MODELS[hparams.model_name](hparams=get_model_params(task, hparams.model_name, hparams.model_config),
                                     cl_token=cl_token)

  trainer_params = get_train_params(hparams.train_config)

  log_dir = os.path.join(hparams.logdir, task.name,
                         hparams.prefix+"_"+model.model_name + "_" + str(hparams.model_config) + "_" + str(
                           trainer_params.learning_rate) + "_" + hparams.exp_name)
  ckpt_dir = os.path.join(hparams.chkpt_dir, task.name,
                          hparams.prefix+"_"+model.model_name + "_" + str(hparams.model_config) + "_" + str(
                            trainer_params.learning_rate) + "_" + hparams.exp_name)
  print(log_dir)
  trainer = Trainer(task=task,
                    model=model,
                    train_params=trainer_params,
                    log_dir=log_dir,
                    ckpt_dir=ckpt_dir)
  trainer.restore()

  distance_hits, distance_total, diff_hits, diff_total = evaluate_vp(trainer.model, trainer.task)
  compute_and_print_acc_stats(distance_hits, distance_total, diff_hits, diff_total)


if __name__ == '__main__':
  app.run(main)
