from distill.distill_util import DistillLoss, get_probs
from tasks.task import Task
import tensorflow as tf
import tensorflow_datasets as tfds

from tf2_models.metrics import ClassificationLoss


class SmallNorb(Task):
  def __init__(self, task_params, name='smallnorb', data_dir='smallnorb'):
    self.databuilder = tfds.builder("smallnorb")
    super(SmallNorb, self).__init__(task_params=task_params, name=name,
                                data_dir=data_dir,
                                builder_cls=None)

  def input_shape(self):
    """
      To be used when calling model.build(input_shape)
    :return:
      #[batch_size, height, width, channels
    """
    return [None, 96, 96, 1]

  def vocab_size(self):
    return 96*96

  def output_size(self):
    return 5

  def get_loss_fn(self):
    return ClassificationLoss(global_batch_size=self.task_params.batch_size,
                              padding_symbol=tf.constant(-1, dtype=tf.int64))

  def get_distill_loss_fn(self, distill_params):
    return DistillLoss(tmp=distill_params.distill_temp)

  def get_probs_fn(self):
    return get_probs

  def metrics(self):
    return [ClassificationLoss(global_batch_size=self.task_params.batch_size,
                              padding_symbol=tf.constant(-1, dtype=tf.int64)),
            tf.keras.metrics.SparseCategoricalAccuracy()]

  @property
  def padded_shapes(self):
    # To make sure we are not using this!
    raise NotImplementedError

  def convert_examples(self, examples):
    return tf.cast(examples['image'], dtype=tf.float32), tf.cast(examples['label_category'], dtype=tf.int32)

  def setup_datasets(self):
    self.info = self.databuilder.info
    self.n_train_batches = int(
      self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(
      self.info.splits['test'].num_examples / self.task_params.batch_size)
    self.n_valid_batches = int(
      self.info.splits['test'].num_examples / self.task_params.batch_size)

    self.databuilder.download_and_prepare(download_dir=self.data_dir)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    assert isinstance(self.test_dataset, tf.data.Dataset)
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.batch(
      batch_size=self.task_params.batch_size)
    self.test_dataset = self.test_dataset.prefetch(
      tf.data.experimental.AUTOTUNE)

    self.train_dataset = self.databuilder.as_dataset(split="train")
    assert isinstance(self.train_dataset, tf.data.Dataset)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.shuffle(1024)
    self.train_dataset = self.train_dataset.batch(
      batch_size=self.task_params.batch_size)
    # self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.prefetch(
      tf.data.experimental.AUTOTUNE)

    self.valid_dataset = self.databuilder.as_dataset(split="test")
    assert isinstance(self.valid_dataset, tf.data.Dataset)
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.batch(
      batch_size=self.task_params.batch_size)
    self.valid_dataset = self.valid_dataset.prefetch(
      tf.data.experimental.AUTOTUNE)

