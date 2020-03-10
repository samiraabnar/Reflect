from distill.distill_util import get_probs
from tasks.task import Task
from tfds_data.bowman_logic import BowmanLogic
from util import constants
import tensorflow as tf

from util.config_util import get_task_params


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