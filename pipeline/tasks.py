import tensorflow as tf
from tf2_models.metrics import masked_sequence_loss
from tf2_models import metrics
from tfds_data.tal_agreement import SVAgreement


class task(object):
  def __init__(self, task_params, name='abstract_task'):
    self.name = name
    self.task_params = task_params

  def convert_examples(self, examples):
    raise NotImplementedError

class SvAgreementLM(task):
  def __init__(self, task_params, name='sv_agreement_lm'):
    super(SvAgreementLM, self).__init__(task_params, name)

    self.databuilder = SVAgreement(data_dir='data')
    self.setup_datasets()

  def setup_datasets(self):
    #with tf.device('/cpu:0'):

    self.valid_dataset = self.databuilder.as_dataset(split="validation", batch_size=self.task_params.batch_size)
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=2)
    self.valid_dataset = self.valid_dataset.prefetch(1)
    #self.test_dataset = self.databuilder.as_dataset(split="test", batch_size=self.task_params.batch_size)
    self.train_dataset = self.databuilder.as_dataset(split="train", batch_size=self.task_params.batch_size)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=2)
    self.train_dataset = self.train_dataset.shuffle(1000000)
    self.train_dataset = self.train_dataset.prefetch(1)
    self.train_dataset = self.train_dataset.cache()

  @tf.function
  def convert_examples(self, examples):
    return examples['sentence'][:,:-1],\
           examples['sentence'][:,1:]

  def get_loss_fn(self):
    return masked_sequence_loss

  @property
  def metrics(self):
    return {'loss': self.get_loss_fn(),
            'accuracy': metrics.accuracy,
            'accuracy_top2': metrics.accuracy_top2,
            'accuracy_top5': metrics.accuracy_top5
          }