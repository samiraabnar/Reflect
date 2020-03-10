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
                                builder_cls=WikiEn)

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    s_shape = tf.shape(sentences)
    # batch_size, length = s_shape[0], s_shape[1]
    bos = self.databuilder.sentence_encoder().encode(constants.bos)
    eos = self.databuilder.sentence_encoder().encode(constants.eos)

    sentence = tf.concat([bos, sentences, eos], axis=-1)
    return sentence[:-1], \
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


if __name__ == '__main__':
  task_params = get_task_params()
  task_params.batch_size = 1
  task = WikiLM()


