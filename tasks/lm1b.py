from tensorflow_datasets.text import Lm1bConfig

from distill.distill_util import get_masked_probs, SequenceDistillLoss
from tasks.tasks import Task
import tensorflow_datasets as tfds
import tensorflow as tf

from tf2_models import metrics
from tf2_models.metrics import MaskedSequenceLoss, masked_batch_perplexity, masked_perplexity
from util import constants


class Lm1B(Task):
  def __init__(self, task_params, name='lm1b', data_dir='data', builder_cls=tfds.text.lm1b.Lm1b):
    super(Lm1B, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=None)

    text_encoder_config = tfds.features.text.TextEncoderConfig(
      encoder_cls=tfds.features.text.SubwordTextEncoder,
      vocab_size=2 ** 13)
    config = Lm1bConfig(
      old_version='1.0.0',
      text_encoder_config=text_encoder_config,
      name='subwords'
    )
    self.databuilder = tfds.text.lm1b.Lm1b(data_dir='data',
                                      config=config)

  def setup_datasets(self):
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.valid_dataset = self.databuilder.as_dataset(split="test")
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                       padded_shapes=self.info.features.shape)
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(10000)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['text']
    s_shape = tf.shape(sentences)
    # batch_size, length = s_shape[0], s_shape[1]
    bos = self.databuilder.sentence_encoder().encode(constants.bos)
    eos = self.databuilder.sentence_encoder().encode(constants.eos)

    sentence = tf.concat([bos, sentences, eos], axis=-1)
    return sentence[:-1], \
           sentence[1:]

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


if __name__ == '__main__':
