from distill.distill_util import DistillLoss, get_probs
from tasks.tasks import Task
import tensorflow as tf
import tensorflow_datasets as tfds

from tf2_models.metrics import ClassificationLoss

class Mnist(Task):
  def __init__(self, task_params, name='mnis', data_dir='mnist_data'):
    self.databuilder = tfds.builder("mnist")
    super(Mnist, self).__init__(task_params=task_params, name=name,
                                data_dir=data_dir,
                                builder_cls=None)

  def output_size(self):
    return 10

  def get_loss_fn(self):
    return ClassificationLoss()

  def get_distill_loss_fn(self, distill_params):
    return DistillLoss(tmp=distill_params.distill_temp)

  def get_probs_fn(self):
    return get_probs

  def metrics(self):
    return [ClassificationLoss(),
            tf.keras.metrics.SparseCategoricalAccuracy()]

  @property
  def padded_shapes(self):
    # To make sure we are not using this!
    raise NotImplementedError

  def setup_datasets(self):
    self.info = self.databuilder.info
    self.n_train_batches = int(
      self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(
      self.info.splits['test'].num_examples / self.task_params.batch_size)
    self.databuilder.download_and_prepare(download_dir=self.data_dir)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    assert isinstance(self.test_dataset, tf.data.Dataset)
    self.train_dataset = self.train_dataset.repeat()
    self.test_dataset = self.test_dataset.batch(
      batch_size=self.task_params.batch_size)
    self.test_dataset = self.test_dataset.prefetch(
      tf.data.experimental.AUTOTUNE)

    self.train_dataset = self.databuilder.as_dataset(split="train")
    assert isinstance(self.train_dataset, tf.data.Dataset)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.shuffle(1024)
    self.train_dataset = self.train_dataset.batch(
      batch_size=self.task_params.batch_size)
    # self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.prefetch(
      tf.data.experimental.AUTOTUNE)
