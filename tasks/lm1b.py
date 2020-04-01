import functools

from tensorflow_datasets.text import Lm1bConfig

from distill.distill_util import get_masked_probs, SequenceDistillLoss
import tensorflow_datasets as tfds
import tensorflow as tf

from tasks.task import Task
from tf2_models import metrics
from tf2_models.metrics import MaskedSequenceLoss, masked_batch_perplexity, masked_perplexity
from util import constants
from util.config_util import get_task_params


class Lm1B(Task):
  def __init__(self, task_params, name='lm1b', data_dir='data', builder_cls=tfds.text.lm1b.Lm1b):
    text_encoder_config = tfds.features.text.TextEncoderConfig(
      encoder_cls=tfds.features.text.SubwordTextEncoder,
      vocab_size=2 ** 13)
    config = Lm1bConfig(
      old_version='1.0.0',
      text_encoder_config=text_encoder_config,
      name='subwords'
    )
    self.databuilder = builder_cls(data_dir=data_dir,
                                   config=config)
    super(Lm1B, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=None)


  def setup_datasets(self):
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.filter(lambda x,y: len(x) < 300)
    self.test_dataset = self.test_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                       padded_shapes=self.padded_shapes,
                                                       padding_values=(self.input_padding_symbol, self.output_padding_symbol))
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.valid_dataset = self.databuilder.as_dataset(split="test")
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.filter(lambda x,y: len(x) < 300)
    self.valid_dataset = self.valid_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                         padded_shapes=self.padded_shapes,
                                                         padding_values=(self.input_padding_symbol, self.output_padding_symbol))
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(10000)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.filter(lambda x,y: len(x) < 300)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                         padded_shapes=self.padded_shapes,
                                                         padding_values=(self.input_padding_symbol, self.output_padding_symbol))
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  def vocab_size(self):
    return self.databuilder.info.features['text'].vocab_size

  def output_size(self):
    return self.vocab_size()

  def sentence_encoder(self):
    return self.databuilder.info.features['text'].encoder

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['text']
    bos = self.databuilder.info.features['text'].encoder.encode(constants.bos)
    eos = self.databuilder.info.features['text'].encoder.encode(constants.eos)

    sentence = tf.concat([bos, sentences, eos], axis=-1)
    return sentence[:-1], \
           sentence[1:]

  def get_loss_fn(self):
    return MaskedSequenceLoss(padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64),
                              num_replicas_in_sync=self.task_params.num_replicas_in_sync)

  def get_distill_loss_fn(self, distill_params):
    return SequenceDistillLoss(tmp=distill_params.distill_temp, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64))

  def get_probs_fn(self):
    return get_masked_probs

  def metrics(self):
    return [MaskedSequenceLoss(padding_symbol=tf.constant(tf.constant(self.output_padding_symbol, dtype=tf.int64), dtype=tf.int64)),
            functools.update_wrapper(functools.partial(masked_batch_perplexity, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64)),
                                     masked_batch_perplexity),
            functools.update_wrapper(functools.partial(masked_perplexity, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64)),
                                     masked_perplexity),
            metrics.AccuracyTopk(global_batch_size=self.task_params.batch_size, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64), topk=1),
            metrics.AccuracyTopk(global_batch_size=self.task_params.batch_size, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64), topk=2),
            metrics.AccuracyTopk(global_batch_size=self.task_params.batch_size, padding_symbol=tf.constant(self.output_padding_symbol, dtype=tf.int64), topk=5)
          ]



if __name__ == '__main__':
  task = Lm1B(get_task_params())

  for x, y in task.valid_dataset:
    print(x[0])
    print(y[0])
    break
