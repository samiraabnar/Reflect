from distill.distill_util import DistillLoss, get_probs
from distill.tasks.tasks import Task
import tensorflow as tf
import tensorflow_datasets as tfds
import os

from tf2_models.metrics import ClassificationLoss


class SST2(Task):
  def __init__(self, task_params, name='sst2', data_dir='data'):
    super(SST2, self).__init__(task_params=task_params, name=name,
                                data_dir=data_dir,
                                builder_cls=tfds.builder("glue/sst2"))
    self.text_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(self.data_dir,'txtencoder'))

  def vocab_size(self):
    return 28*28

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
    return ([None],[])

  def convert_examples(self, examples):
    return self.text_encoder.encode(examples['sentence']), examples['label']




if __name__ == '__main__':
  examples, metadata = tfds.load('glue/sst2')
  train_examples, val_examples = examples['train'], examples['validation']

  txt_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (example['sentence'].numpy() for example in train_examples), target_vocab_size=2 ** 13)
  txt_encoder.save_to_file('data/glue/tokenizer')


