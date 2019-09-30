import tensorflow as tf
from tf2_models.metrics import masked_sequence_loss
from tf2_models import metrics
from tfds_data.tal_agreement import SVAgreement


class task(object):
  def __init__(self, task_params, name='abstract_task', data_dir='data'):
    self.name = name
    self.task_params = task_params
    self.data_dir = data_dir

  def convert_examples(self, examples):
    raise NotImplementedError

class SvAgreementLM(task):
  def __init__(self, task_params, name='sv_agreement_lm', data_dir='data'):
    super(SvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir='data')

    self.databuilder = SVAgreement(data_dir='data')
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['validation'].num_examples / task_params.batch_size)
    self.setup_datasets()


  def setup_datasets(self):
    #with tf.device('/cpu:0'):

    self.valid_dataset = self.databuilder.as_dataset(split="validation")
    self.valid_dataset = self.valid_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.valid_dataset = self.valid_dataset.cache()
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    #self.test_dataset = self.databuilder.as_dataset(split="test")
    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(self.info.splits['train'].num_examples)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


  @tf.function
  def convert_examples(self, examples):
    return examples['sentence'][:,:-1],\
           examples['sentence'][:,1:]

  def get_loss_fn(self):
    return tf.losses.sparse_categorical_crossentropy

  @property
  def metrics(self):
    return {'loss': self.get_loss_fn(),
            'accuracy': metrics.accuracy,
            'accuracy_top2': metrics.accuracy_top2,
            'accuracy_top5': metrics.accuracy_top5
          }