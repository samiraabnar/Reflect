from distill.distill_util import DistillLoss, get_probs, SequenceDistillLoss, get_topk_masked_probs, get_masked_probs
from tasks.task import Task
import tensorflow as tf

from tf2_models import metrics
from tf2_models.metrics import masked_batch_perplexity, masked_perplexity, \
  MaskedSequenceLoss, ClassificationLoss
from tfds_data.tal_agreement import WordSvAgreement, SVAgreement
from util import constants


class SvAgreementLM(Task):
  def __init__(self, task_params, name='sv_agreement_lm', data_dir='data', builder_cls=SVAgreement):
    super(SvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    s_shape = tf.shape(sentences)
    #batch_size, length = s_shape[0], s_shape[1]
    bos =  self.databuilder.sentence_encoder().encode(constants.bos)
    eos =  self.databuilder.sentence_encoder().encode(constants.eos)

    sentence = tf.concat([bos, sentences, eos], axis=-1)
    return sentence[:-1],\
           sentence[1:]

  def get_loss_fn(self):
    return MaskedSequenceLoss(padding_symbol=0)

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

  def get_distill_loss_fn(self, distill_params):
    return SequenceDistillLoss(tmp=distill_params.distill_temp, padding_symbol=0)

  def get_probs_fn(self):
    return get_masked_probs

  def metrics(self):
    return [MaskedSequenceLoss(padding_symbol=0),
            masked_batch_perplexity,
            masked_perplexity,
            metrics.accuracy,
            metrics.accuracy_top2,
            metrics.accuracy_top5
          ]


class WordSvAgreementLM(SvAgreementLM):
  def __init__(self, task_params, name='word_sv_agreement_lm', data_dir='data', builder_cls=WordSvAgreement):
    super(WordSvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)


class WordSvAgreementVP(Task):
  def __init__(self, task_params, name='word_sv_agreement_vp', data_dir='data', builder_cls=WordSvAgreement):
    super(WordSvAgreementVP, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)
    self.output_padding_symbol = -1
  @property
  def padded_shapes(self):
    return ([None],[])

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    bos = self.databuilder.sentence_encoder().encode(constants.bos)
    eos = self.databuilder.sentence_encoder().encode(constants.eos)

    sentences = tf.concat([bos, sentences, eos], axis=-1)

    verb_position = examples['verb_position']+1  #+1 because of adding bos.

    # The verb it self is also masked
    mask = tf.cast(tf.sequence_mask(verb_position,maxlen=tf.shape(sentences)[0]), dtype=tf.int64)
    max_length = tf.reduce_max(verb_position + 1)

    last_index_mask = tf.eye(tf.shape(sentences)[0], dtype=tf.int64)[verb_position]
    last_index_mask = last_index_mask * eos[0]

    return (sentences * mask + last_index_mask)[:max_length], \
           examples['verb_class']

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return 2

  def get_loss_fn(self):
    return ClassificationLoss(padding_symbol=-1)

  def get_distill_loss_fn(self, distill_params):
    return DistillLoss(tmp=distill_params.distill_temp, padding_symbol=-1)

  def get_probs_fn(self):
    return get_probs


  def metrics(self):
    return [ClassificationLoss(padding_symbol=-1),
            tf.keras.metrics.SparseCategoricalAccuracy()]
