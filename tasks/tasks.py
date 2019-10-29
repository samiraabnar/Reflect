import tensorflow as tf
from distill.distill_util import get_masked_probs


class Task(object):
  def __init__(self, task_params, builder_cls, name='abstract_task', data_dir='data'):
    self.name = name
    self.output_padding_symbol = 0
    self.task_params = task_params
    self.data_dir = data_dir
    self.builder_cls = builder_cls
    self.databuilder = self.builder_cls(data_dir=self.data_dir)

    self.setup_datasets()

  @property
  def padded_shapes(self):
    return ([None],[None])

  def vocab_size(self):
    raise NotImplementedError

  def convert_examples(self, examples):
    raise NotImplementedError

  def get_probs_fn(self):
    return get_masked_probs

  def setup_datasets(self):
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['validation'].num_examples / self.task_params.batch_size)

    self.valid_dataset = self.databuilder.as_dataset(split="validation")
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.padded_shapes)
    #self.valid_dataset = self.valid_dataset.cache()
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                         padded_shapes=self.padded_shapes)
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(10000)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.padded_shapes)
    #self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)



