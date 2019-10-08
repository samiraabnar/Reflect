import tensorflow as tf
from tf2_models.metrics import masked_sequence_loss
from tf2_models import metrics
from tfds_data.tal_agreement import SVAgreement, WordSvAgreement
from util.config_util import get_task_params


class task(object):
  def __init__(self, task_params, builder_cls, name='abstract_task', data_dir='data'):
    self.name = name
    self.task_params = task_params
    self.data_dir = data_dir
    self.builder_cls = builder_cls

    self.databuilder = self.builder_cls(data_dir=self.data_dir)
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / task_params.batch_size)
    self.n_valid_batches = int(self.info.splits['validation'].num_examples / task_params.batch_size)
    self.setup_datasets()

  def vocab_size(self):
    raise NotImplementedError

  def convert_examples(self, examples):
    raise NotImplementedError

  def setup_datasets(self):
    #with tf.device('/cpu:0'):

    self.valid_dataset = self.databuilder.as_dataset(split="validation")
    self.valid_dataset = self.valid_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.valid_dataset = self.valid_dataset.cache()
    self.valid_dataset = self.valid_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.valid_dataset = self.valid_dataset.repeat()
    self.valid_dataset = self.valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    self.test_dataset = self.databuilder.as_dataset(split="test")
    self.test_dataset = self.test_dataset.padded_batch(batch_size=self.task_params.batch_size,
                                                         padded_shapes=self.info.features.shape)
    # self.test_dataset = self.test_dataset.cache()
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    #self.test_dataset = self.databuilder.as_dataset(split="test")
    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(self.info.splits['train'].num_examples)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)



class SvAgreementLM(task):
  def __init__(self, task_params, name='sv_agreement_lm', data_dir='data', builder_cls=SVAgreement):
    super(SvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)

  @tf.function
  def convert_examples(self, examples):
    return examples['sentence'][:,:-1],\
           examples['sentence'][:,1:]

  def get_loss_fn(self):
    return masked_sequence_loss

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

  @property
  def metrics(self):
    return {'loss': self.get_loss_fn(),
            'accuracy': metrics.accuracy,
            'accuracy_top2': metrics.accuracy_top2,
            'accuracy_top5': metrics.accuracy_top5
          }


class WordSvAgreementLM(SvAgreementLM):
  def __init__(self, task_params, name='word_sv_agreement_lm', data_dir='data', builder_cls=WordSvAgreement):
    super(WordSvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)


class WordSvAgreementVP(task):
  def __init__(self, task_params, name='word_sv_agreement_vp', data_dir='data', builder_cls=WordSvAgreement):
    super(WordSvAgreementVP, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    mask = tf.cast(tf.sequence_mask(examples['verb_position'],maxlen=tf.shape(sentences)[1]), dtype=tf.int64)
    return sentences * mask, \
           examples['verb_class']

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return 2

  def get_loss_fn(self):
    return masked_sequence_loss


if __name__ == '__main__':
    task = WordSvAgreementVP(get_task_params())

    x, y = iter(task.valid_dataset).next()
    print(x)
    print(y)
