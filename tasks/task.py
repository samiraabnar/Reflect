import tensorflow as tf
from distill.distill_util import get_masked_probs
from distill.repsim_util import rep_loss
from util import constants


class Task(object):
  def __init__(self, task_params, num_replicas_in_sync=1, builder_cls=None, name='abstract_task', data_dir='data'):
    self.name = name
    self.output_padding_symbol = 0
    self.task_params = task_params
    self.data_dir = data_dir
    self.builder_cls = builder_cls
    self.num_replicas_in_sync = num_replicas_in_sync
    if builder_cls:
      self.databuilder = self.builder_cls(data_dir=self.data_dir)

    self.input_padding_symbol = tf.cast(self.sentence_encoder().encode(constants.pad)[0], dtype=tf.int64)
    self.output_padding_symbol = None
    self.setup_datasets()



  def sentence_encoder(self):
    raise NotImplementedError
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
    assert self.databuilder
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['validation'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)


    self.valid_dataset = self.databuilder.as_dataset(split="validation")
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                         padded_shapes=self.padded_shapes,
                                                         padding_values=(self.input_padding_symbol,self.output_padding_symbol))
    #self.valid_dataset = self.valid_dataset.cache()
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                       padded_shapes=self.padded_shapes,
                                                       padding_values=(self.input_padding_symbol,self.output_padding_symbol))
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(10000)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                         padded_shapes=self.padded_shapes,
                                                         padding_values=(self.input_padding_symbol,self.output_padding_symbol))
    #self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  def get_rep_loss(self):
    return rep_loss


class RandomGaussianTask(object):
  def __init__(self, task_params, builder_cls=None, name='random_gaussian_task', data_dir='data'):
    self.name = name
    self.output_padding_symbol = 0
    self.task_params = task_params
    self.data_dir = data_dir
    self.builder_cls = builder_cls
    if builder_cls:
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
    assert self.builder_cls
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['validation'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)


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
