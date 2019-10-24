import tensorflow as tf
import tensorflow_datasets as tfds

from distill.distill_util import DistillLoss, get_probs, get_masked_probs
from tf2_models.metrics import masked_sequence_loss, sequence_loss, masked_perplexity, masked_batch_perplexity, \
  batch_masked_sequence_loss
from tf2_models import metrics
from tfds_data.bowman_logic import BowmanLogic
from tfds_data.tal_agreement import SVAgreement, WordSvAgreement
from util import constants
from util.config_util import get_task_params


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


    #with tf.device('/cpu:0'):

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



class Lm1B(Task):
  def __init__(self, task_params, name='lm1b', data_dir='data', builder_cls=tfds.text.lm1b.Lm1b):
    super(Lm1B, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)

  def setup_datasets(self):
    self.info = self.databuilder.info
    self.n_train_batches = int(self.info.splits['train'].num_examples / self.task_params.batch_size)
    self.n_test_batches = int(self.info.splits['test'].num_examples / self.task_params.batch_size)

    #with tf.device('/cpu:0'):

    self.test_dataset = self.databuilder.as_dataset(split="test")
    self.test_dataset = self.test_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.test_dataset = self.test_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.valid_dataset = self.valid_dataset.cache()
    self.test_dataset = self.test_dataset.repeat()
    self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)



    #self.test_dataset = self.databuilder.as_dataset(split="test")
    self.train_dataset = self.databuilder.as_dataset(split="train")
    self.train_dataset = self.train_dataset.shuffle(10000)
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

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
    self.train_dataset = self.train_dataset.shuffle(10000)
    self.train_dataset = self.train_dataset.padded_batch(batch_size=self.task_params.batch_size, padded_shapes=self.info.features.shape)
    #self.train_dataset = self.train_dataset.cache()
    self.train_dataset = self.train_dataset.map(map_func=lambda x: self.convert_examples(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


class SvAgreementLM(Task):
  def __init__(self, task_params, name='sv_agreement_lm', data_dir='data', builder_cls=SVAgreement):
    super(SvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)

  # @tf.function
  # def convert_examples(self, examples):
  #   sentences = examples['sentence']
  #   s_shape = tf.shape(sentences)
  #   batch_size, length = s_shape[0], s_shape[1]
  #   bos = tf.ones((batch_size,1), dtype=tf.int64) * self.databuilder.sentence_encoder().encode(constants.bos)
  #   eos = tf.ones((batch_size,1), dtype=tf.int64) * self.databuilder.sentence_encoder().encode(constants.eos)
  #
  #   sentence = tf.concat([bos, sentences, eos], axis=1)
  #   return sentence[:,:-1],\
  #          sentence[:,1:]

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    s_shape = tf.shape(sentences)
    #batch_size, length = s_shape[0], s_shape[1]
    bos =  self.databuilder.sentence_encoder().encode(constants.bos)
    eos =  self.databuilder.sentence_encoder().encode(constants.eos)

    sentence = tf.concat([bos, sentences, eos], axis=-1)
    return sentence[:-1],\
           sentence[1:]

  def get_loss_fn(self):
    return batch_masked_sequence_loss

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

  def metrics(self):
    return [self.get_loss_fn(),
            masked_batch_perplexity,
            masked_perplexity,
            metrics.accuracy,
            metrics.accuracy_top2,
            metrics.accuracy_top5
          ]


class WordSvAgreementLM(SvAgreementLM):
  def __init__(self, task_params, name='word_sv_agreement_lm', data_dir='data', builder_cls=WordSvAgreement):
    super(WordSvAgreementLM, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)


class WordSvAgreementVP(Task):
  def __init__(self, task_params, name='word_sv_agreement_vp', data_dir='data', builder_cls=WordSvAgreement):
    super(WordSvAgreementVP, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)
    self.output_padding_symbol = -1
  @property
  def padded_shapes(self):
    return ([None],[])

  @tf.function
  def convert_examples(self, examples):
    sentences = examples['sentence']
    bos = self.databuilder.sentence_encoder().encode(constants.bos)
    eos = self.databuilder.sentence_encoder().encode(constants.eos)

    sentences = tf.concat([bos, sentences, eos], axis=-1)

    verb_position = examples['verb_position']+1  #+1 because of adding bos.

    # The verb it self is also masked
    mask = tf.cast(tf.sequence_mask(verb_position,maxlen=tf.shape(sentences)[0]), dtype=tf.int64)
    max_length = tf.reduce_max(verb_position + 1)

    last_index_mask = tf.eye(tf.shape(sentences)[0], dtype=tf.int64)[verb_position]
    last_index_mask = last_index_mask * eos[0]

    return (sentences * mask + last_index_mask)[:max_length], \
           examples['verb_class']

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return 2

  def get_loss_fn(self):
    return sequence_loss

  def get_distill_loss_fn(self, distill_params):
    return DistillLoss(tmp=distill_params.distill_temp, padding_symbol=-1)

  def metrics(self):
    return [self.get_loss_fn(),
            tf.keras.metrics.SparseCategoricalAccuracy()]


class BowmanLogicConcat(Task):
  def __init__(self, task_params, name='bowman_logic_task', data_dir='data', builder_cls=BowmanLogic):
    super(BowmanLogicConcat, self).__init__(task_params=task_params, name=name, data_dir=data_dir, builder_cls=builder_cls)

  @tf.function
  def convert_examples(self, examples):
    statement_a = examples['statement_a']
    statement_b = examples['statement_a']

    bos =  self.databuilder.sentence_encoder().encode(constants.bos)
    eos =  self.databuilder.sentence_encoder().encode(constants.eos)
    tf.print(bos, eos)
    sentence = tf.concat([bos, statement_a, bos, statement_b, eos], axis=-1)
    return sentence,\
           examples['relation']

  @property
  def padded_shapes(self):
    return ([None],[])

  def get_loss_fn(self):
    return sequence_loss

  def vocab_size(self):
    return self.databuilder.vocab_size()

  def output_size(self):
    return self.vocab_size()

  def metrics(self):
    return [self.get_loss_fn(),
            tf.keras.metrics.SparseCategoricalAccuracy()
          ]

  def get_probs_fn(self):
    return get_probs

if __name__ == '__main__':
    task = BowmanLogicConcat(get_task_params())

    x, y = iter(task.valid_dataset).next()
    print(x[0])
    print(y[0])
    print(task.databuilder.sentence_encoder().decode(x[0][:10]))
    print(task.databuilder.info)
