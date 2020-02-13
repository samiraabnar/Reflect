import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from util import constants
from collections import Counter
from util.models import MODELS
from util.tasks import TASKS
from util.config_util import get_model_params, get_task_params, get_train_params

from util import inflect
dependency_fields = ['sentence', 'orig_sentence', 'pos_sentence',
                     'subj', 'verb', 'subj_pos', 'has_rel', 'has_nsubj',
                     'verb_pos', 'subj_index', 'verb_index', 'n_intervening',
                     'last_intervening', 'n_diff_intervening', 'distance',
                     'max_depth', 'all_nouns', 'nouns_up_to_verb']


def get_model(config, task, hparams, cl_token):
    model = MODELS[config['model_name']](hparams=hparams, cl_token=cl_token)


    ckpt_dir = os.path.join(config['chkpt_dir'],task.name,
                            model.model_name+"_"+str(config['model_config'])+"_"+str(config['learning_rate'])+"_"+config['exp_name'])

    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored student from {}".format(manager.latest_checkpoint))
    else:
      print("No checkpoint found {}".format(ckpt_dir))

    model.compile(loss=task.get_loss_fn(), metrics=task.metrics())
    
    
    return model, ckpt


def get_student_model(config, task, hparams, cl_token):
    teacher_model = MODELS[config['teacher_model']](hparams=get_model_params(task, config['teacher_model'], config['teacher_config']), cl_token=cl_token)
    model = MODELS[config['student_model']](hparams=hparams, cl_token=cl_token)


    ckpt_dir = os.path.join(config['chkpt_dir'], task.name,
                              '_'.join([config['distill_mode'],config['distill_config'],
                                        "teacher", teacher_model.model_name, 
                                        config['teacher_config'],
                                        config['teacher_exp_name'],
                                       "student",model.model_name,
                                        str(config['student_config']),
                                        config['student_exp_name']]))
    print("student_checkpoint:", ckpt_dir)

    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored student from {}".format(manager.latest_checkpoint))
    else:
      print("No checkpoint found {}".format(ckpt_dir))

    model.compile(loss=task.get_loss_fn(), metrics=task.metrics())
    
    
    return model, ckpt



def gen_inflect_from_vocab(infl_eng, vocab_file, freq_threshold=1000):
    vbp = {}
    vbz = {}
    nn = {}
    nns = {}
    from_pos = {'NNS': nns, 'NN': nn, 'VBP': vbp, 'VBZ': vbz}

    for line in open(vocab_file):
        if line.startswith(' '):   # empty string token
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



def compute_and_print_acc_stats(distance_hits, distance_total, diff_hits, diff_total):
  ''' Computes and prints accuracy based on hits

  :param distance_hits:
  :param distance_total:
  :param diff_hits:
  :param diff_total:
  :return: None
  '''
  dis_acc = np.zeros(16)
  dif_acc = np.zeros(5)
  total_nominator = 0.0
  total_denominator = 0.0
  print('Accuracy by distance')
  for k in sorted(distance_hits.keys()):
    v = distance_hits[k]
    acc = v / distance_total[k]
    dis_acc[k-1] = acc
    print("%d | %.2f" % (k, acc), distance_total[k])
    total_nominator += v
    total_denominator += distance_total[k]

  print("Micro accuracy (distance):", total_nominator / total_denominator)
  print("Macro accuracy (distance):", np.mean(dis_acc))

  print('Accuracy by intervenings:')
  total_nominator = 0.0
  total_denominator = 0.0
  for k in sorted(diff_hits.keys()):
    v = diff_hits[k]
    acc = v * 1. / diff_total[k]
    print("%d | %.2f" % (k, acc), diff_total[k])
    dif_acc[k] = acc
    total_nominator += v
    total_denominator += diff_total[k]

  print("Micro accuracy (intervenings):", total_nominator / total_denominator)
  print("Macro accuracy (intervenings):", np.mean(dif_acc))

    
def evaluate_vp_cl(model, verb_infl, noun_infl, task, split='test', batch_size=1000):
    distance_hits = Counter()
    distance_total = Counter()
    diff_hits = Counter()
    diff_total = Counter()

    test_data = task.databuilder.as_dataset(split=split, batch_size=batch_size)
    e = 0
    for examples in tqdm(test_data):
        e += 1
        sentences = examples['sentence']
        bos = tf.cast(task.databuilder.sentence_encoder().encode(constants.bos) * tf.ones((sentences.shape[0],1)), dtype=tf.int64)
        eos = tf.cast(task.databuilder.sentence_encoder().encode(constants.eos) *tf.ones((sentences.shape[0],1)), dtype=tf.int64)

        sentences = tf.concat([bos, sentences, eos], axis=-1)

        verb_position = examples['verb_position']+1  #+1 because of adding bos.
        # The verb it self is also masked
        mask = tf.cast(tf.sequence_mask(verb_position,maxlen=tf.shape(sentences)[1]), dtype=tf.int64)
        max_length = tf.reduce_max(verb_position + 1)

        last_index_mask = tf.gather(tf.eye(tf.shape(sentences)[1], dtype=tf.int64),verb_position)
        last_index_mask = last_index_mask * eos[0]

        inputs = (sentences * mask + last_index_mask)[:,:max_length]


        s_shape = tf.shape(inputs)
        batch_size, length = s_shape[0], s_shape[1]
        verb_classes = examples['verb_class']
        actual_verbs = examples['verb']
        inflected_verbs = [verb_infl[v.decode("utf-8")] for v in actual_verbs.numpy()]

        distances = examples['distance'].numpy()
        nz = examples['n_intervening'].numpy()
        n_diffs = examples['n_diff_intervening'].numpy()

        actual_verb_indexes = [task.databuilder.sentence_encoder().encode(v)[0] for v in actual_verbs.numpy()]
        inflected_verb_indexes = [task.databuilder.sentence_encoder().encode(v)[0] for v in inflected_verbs]


        predictions = model(inputs)
        predictions = np.argmax(predictions, axis=-1)
        corrects = predictions == verb_classes

        for i, c in enumerate(corrects):
            if actual_verb_indexes[i] == 10035:
                continue
            if nz[i] > 4 or distances[i] > 16:
                continue

            distance_total[distances[i]] += 1
            distance_hits[distances[i]] += int(c)
            if nz[i] == n_diffs[i]:
                n = nz[i]
                diff_total[n] += 1
                diff_hits[n] += int(c)
    
    return distance_hits, distance_total, diff_hits, diff_total