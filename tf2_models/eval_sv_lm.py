import os
from tasks.tasks import SvAgreementLM, WordSvAgreementLM, WordSvAgreementVP
from tf2_models.lm_transformer import LmGPT2, ClassifierGPT2
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.lm_lstm import LmLSTM, LmLSTMSharedEmb, ClassifierLSTM
from tf2_models.trainer import Trainer
import numpy as np
from util import constants
from collections import Counter
from tqdm import tqdm
from tf2_models.metrics import *

MODELS = {"lm_lstm": LmLSTM,
          "lm_gpt2": LmGPT2,
          "lm_lstm_shared_emb": LmLSTMSharedEmb,
          'cl_gpt2': ClassifierGPT2,
          'cl_lstm': ClassifierLSTM}


log_dir = "logs"
chkpt_dir = "tf_ckpts"
exp_name = "tune_withl2_6"

task = WordSvAgreementLM(task_params=get_task_params(),data_dir='data')


model_config = 'big_gpt_v4'
model_name = 'lm_gpt2'
train_config ='adam_slow'
# Create the Model
model_params = get_model_params(task,model_name, model_config)
print("model_params: ", model_params.__dict__)

cl_token = task.databuilder.sentence_encoder().encode(constants.bos)
model = MODELS[model_name](hparams=get_model_params(task,model_name, model_config), cl_token=cl_token)

trainer_params = get_train_params(train_config)

log_dir = os.path.join(log_dir,task.name, model.model_name+"_"+str(model_config)+"_"+str(trainer_params.learning_rate)+"_"+exp_name)
ckpt_dir = os.path.join(chkpt_dir,task.name, model.model_name+"_"+str(model_config)+"_"+str(trainer_params.learning_rate)+"_"+exp_name)

print(log_dir)

trainer = Trainer(task=task,
                model=model,
                train_params=trainer_params,
                log_dir=log_dir,
                ckpt_dir=ckpt_dir)

trainer.restore()


def gen_inflect_from_vocab(vocab_file, freq_threshold=1000):
  vbp = {}
  vbz = {}
  nn = {}
  nns = {}
  from_pos = {'NNS': nns, 'NN': nn, 'VBP': vbp, 'VBZ': vbz}

  for line in open(vocab_file):
    if line.startswith(' '):  # empty string token
      continue
    word, pos, count = line.strip().split()
    count = int(count)
    if len(word) > 1 and pos in from_pos and count >= freq_threshold:
      from_pos[pos][word] = count

  verb_infl = {'VBP': 'VBZ', 'VBZ': 'VBP'}
  for word, count in vbz.items():
    candidate = infl_eng.plural_verb(word)
    if candidate in vbp:
      verb_infl[candidate] = word
      verb_infl[word] = candidate

  noun_infl = {'NN': 'NNS', 'NNS': 'NN'}
  for word, count in nn.items():
    candidate = infl_eng.plural_noun(word)
    if candidate in nns:
      noun_infl[candidate] = word
      noun_infl[word] = candidate

  return verb_infl, noun_infl

from util import inflect

infl_eng = inflect.engine()

dependency_fields = ['sentence', 'orig_sentence', 'pos_sentence',
                     'subj', 'verb', 'subj_pos', 'has_rel', 'has_nsubj',
                     'verb_pos', 'subj_index', 'verb_index', 'n_intervening',
                     'last_intervening', 'n_diff_intervening', 'distance',
                     'max_depth', 'all_nouns', 'nouns_up_to_verb']

verb_infl, noun_infl = gen_inflect_from_vocab('data/tal_agreement/wiki.vocab')

distance_hits = Counter()
distance_total = Counter()
diff_hits = Counter()
diff_total = Counter()

test_data = task.databuilder.as_dataset(split='test', batch_size=1000)
e = 0
for example in tqdm(test_data):
  e += 1
  encoded_sentences = example['sentence']
  s_shape = tf.shape(encoded_sentences)
  batch_size, length = s_shape[0], s_shape[1]
  bos = tf.ones((batch_size, 1), dtype=tf.int64) * task.databuilder.sentence_encoder().encode(constants.bos)
  eos = tf.ones((batch_size, 1), dtype=tf.int64) * task.databuilder.sentence_encoder().encode(constants.eos)

  encoded_sentences = tf.concat([bos, encoded_sentences, eos], axis=1)

  actual_verbs = example['verb']
  inflected_verbs = [verb_infl[v.decode("utf-8")] for v in actual_verbs.numpy()]
  verb_indexes = example['verb_position'] - 1
  distances = example['distance'].numpy()
  nz = example['n_intervening'].numpy()
  n_diffs = example['n_diff_intervening'].numpy()

  sentence = task.databuilder.sentence_encoder().decode(encoded_sentences[0])
  actual_verb_indexes = [task.databuilder.sentence_encoder().encode(v)[0] for v in actual_verbs.numpy()]
  inflected_verb_indexes = [task.databuilder.sentence_encoder().encode(v)[0] for v in inflected_verbs]

  scores = model(encoded_sentences)
  actual_batch_indexes = [(i, verb_indexes[i], actual_verb_indexes[i]) for i in range(len(verb_indexes))]
  actual_scores = tf.compat.v2.gather_nd(scores, actual_batch_indexes)

  inflected_batch_indexes = [(i, verb_indexes[i], inflected_verb_indexes[i]) for i in range(len(verb_indexes))]
  infelected_scores = tf.compat.v2.gather_nd(scores, inflected_batch_indexes)

  corrects = actual_scores > infelected_scores
  for i, c in enumerate(corrects):
    if verb_indexes[i] == 10035:
      continue
    if nz[i] > 4 or distances[i] > 16:
      continue

    distance_total[distances[i]] += 1
    distance_hits[distances[i]] += int(c)
    if nz[i] == n_diffs[i]:
      n = nz[i]
      diff_total[n] += 1
      diff_hits[n] += int(c)

dis_acc = {}
dis_acc = np.zeros(17)
dif_acc = np.zeros(5)
print('Accuracy by distance')
for k in sorted(distance_hits.keys()):
    v = distance_hits[k]
    acc = v / distance_total[k]
    dis_acc[k] = acc
    print("%d | %.2f" % (k, acc), distance_total[k])

print('Accuracy by intervenings')
for k in sorted(diff_hits.keys()):
    v = diff_hits[k]
    acc = v * 1./diff_total[k]
    print("%d | %.2f" % (k, acc), diff_total[k])
    dif_acc[k] = acc

stats = {'distance': dis_acc, 'intervenings': dif_acc}