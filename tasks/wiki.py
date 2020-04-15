import functools

from distill.distill_util import DistillLoss, get_probs, SequenceDistillLoss, get_masked_probs
from tasks.task import Task
import tensorflow as tf
import tensorflow_datasets as tfds

from tf2_models import metrics
from tf2_models.metrics import ClassificationLoss, MaskedSequenceLoss, masked_batch_perplexity, masked_perplexity
from tfds_data.wiki import WikiEn
from util import constants
from util.config_util import get_task_params


class WikiLM(Task):
  def __init__(self, task_params, name='wikilm', data_dir='data'):
    super(WikiLM, self).__init__(task_params=task_params, name=name,
                                data_dir=data_dir,
                                builder_cls=WikiEn,
                                output_padding=True)


  def sentence_encoder(self):
    return self.databuilder.sentence_encoder()

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    bos = self.databuilder.sentence_encoder().encode(constants.bos)
    eos = self.databuilder.sentence_encoder().encode(constants.eos)

    sentence = tf.concat([bos, sentences, eos], axis=-1)
    return sentence[:-1], \
           sentence[1:]

  def get_loss_fn(self):
    return MaskedSequenceLoss(padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64),
                              num_replicas_in_sync=self.task_params.num_replicas_in_sync)

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

  def get_distill_loss_fn(self, distill_params):
    return SequenceDistillLoss(tmp=distill_params.distill_temp, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64))

  def get_probs_fn(self):
    return get_masked_probs

  def metrics(self):
    return [MaskedSequenceLoss(padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64)),
            functools.update_wrapper(functools.partial(masked_batch_perplexity,
                                                       padding_symbol=tf.constant(self.output_padding_symbol,
                                                                                  dtype=tf.int64)),
                                     masked_batch_perplexity),
            functools.update_wrapper(functools.partial(masked_perplexity,
                                                       padding_symbol=tf.constant(self.output_padding_symbol,
                                                                                  dtype=tf.int64)),
                                     masked_perplexity),
            metrics.AccuracyTopk(global_batch_size=self.task_params.batch_size, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64), topk=1),
            metrics.AccuracyTopk(global_batch_size=self.task_params.batch_size, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64), topk=2),
            metrics.AccuracyTopk(global_batch_size=self.task_params.batch_size, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64), topk=5)
          ]


if __name__ == '__main__':
  task_params = get_task_params()
  task_params.batch_size = 1
  task = WikiLM()


