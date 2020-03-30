from distill.distill_util import DistillLoss, get_probs, SequenceDistillLoss, get_masked_probs
from tasks.task import Task
import tensorflow as tf
import tensorflow_datasets as tfds

from tf2_models import metrics
from tf2_models.metrics import ClassificationLoss, ClassificationLossMetric, MaskedSequenceLoss, \
  masked_batch_perplexity, masked_perplexity
from tfds_data.sst2 import SST2


class ClassifySST2(Task):
  def __init__(self, task_params, name='sst2', data_dir='data'):
    super(ClassifySST2, self).__init__(task_params=task_params, name=name,
                                data_dir=data_dir,
                                builder_cls=SST2)

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def sentence_encoder(self):
    return self.databuilder.sentence_encoder()

  def output_size(self):
    return 2

  def get_loss_fn(self):
    return ClassificationLoss(global_batch_size=tf.constant(self.task_params.batch_size, dtype=tf.float32), padding_symbol=tf.constant(-1, dtype=tf.int64))

  def get_distill_loss_fn(self, distill_params):
    return DistillLoss(tmp=distill_params.distill_temp)

  def get_probs_fn(self):
    return get_probs

  def metrics(self):
    return [ClassificationLossMetric(global_batch_size=tf.constant(self.task_params.batch_size, dtype=tf.float32),padding_symbol=tf.constant(-1, dtype=tf.int64)),
            tf.keras.metrics.SparseCategoricalAccuracy()]

  @property
  def padded_shapes(self):
    return ([None],[])

  def convert_examples(self, examples):
    return tf.convert_to_tensor(examples['sentence']), tf.convert_to_tensor(examples['label'])


class LmSST2(Task):
  def __init__(self, task_params, name='lm_sst2', data_dir='data'):
    super(LmSST2, self).__init__(task_params=task_params, name=name,
                                data_dir=data_dir,
                                builder_cls=SST2)

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def sentence_encoder(self):
    return self.databuilder.sentence_encoder()

  def output_size(self):
    return self.vocab_size()

  def get_loss_fn(self):
    return MaskedSequenceLoss(padding_symbol=0)

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


  @property
  def padded_shapes(self):
    return ([None],[None])

  def convert_examples(self, examples):
    return tf.convert_to_tensor(examples['sentence'][:-1]), tf.convert_to_tensor(examples['sentence'][1:])


if __name__ == '__main__':
  examples = tfds.load('glue/sst2')
  train_examples, val_examples = examples['train'], examples['validation']

  txt_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (example['sentence'].numpy() for example in train_examples), target_vocab_size=2 ** 13)
  txt_encoder.save_to_file('data/sst2/tokenizer')


