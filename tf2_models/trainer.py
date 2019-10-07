import absl
import tensorflow as tf
import os

from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay, LearningRateSchedule

from tf2_models.keras_callbacks import CheckpointCallback, SummaryCallback
from tf2_models.metrics import masked_sequence_loss
from tf2_models.train_utils import RectifiedAdam, ExponentialDecayWithWarmpUp

OPTIMIZER_DIC = {'adam': tf.keras.optimizers.Adam,
                 'radam': RectifiedAdam,
                 }
class Trainer(object):

  def __init__(self, model, task, train_params, log_dir, ckpt_dir):
    self.model = model
    self.task = task
    self.train_params = train_params

    if self.train_params.optimizer == 'radam':
      class Constant(LearningRateSchedule):
        def __init__(self, lr):
          self.learning_rate = lr
        def __call__(self, step):
          return self.learning_rate

      lr_schedule = Constant(self.train_params.learning_rate)
    else:
      initial_learning_rate = self.train_params.learning_rate
      lr_schedule = ExponentialDecayWithWarmpUp(
        initial_learning_rate=initial_learning_rate,
        decay_steps=self.train_params.decay_steps,
        decay_rate=0.96,
        warmup_steps=self.train_params.warmup_steps)

    self.optimizer = OPTIMIZER_DIC[self.train_params.optimizer](learning_rate=lr_schedule, epsilon=1e-08, clipnorm=1.0)

    self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
    self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=2)


    x, y = iter(self.task.valid_dataset).next()
    model(x)
    model.summary()

    model.compile(
      optimizer=self.optimizer,
      loss=masked_sequence_loss,
      metrics=[masked_sequence_loss])
    #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),)


    summary_dir = os.path.join(log_dir, 'summaries')
    tf.io.gfile.makedirs(log_dir)
    self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(summary_dir, 'train'))
    tf.compat.v2.summary.experimental.set_step(self.optimizer.iterations)

    ckpt_callback = CheckpointCallback(manager=self.manager, ckpt=self.ckpt)
    summary_callback = SummaryCallback(summary_writer=self.summary_writer)

    self.callbacks = [ckpt_callback, summary_callback]

  def restore(self):
    self.ckpt.restore(self.manager.latest_checkpoint)
    if self.manager.latest_checkpoint:
      print("Restored from {}".format(self.manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")

  def train(self):
    with self.summary_writer.as_default():
      print("initial learning rate:", self.model.optimizer.learning_rate(self.model.optimizer.iterations))
      self.model.fit(self.task.train_dataset,
                epochs=self.train_params.num_train_epochs,
                steps_per_epoch=self.task.n_train_batches,
                validation_steps=self.task.n_valid_batches,
                callbacks=self.callbacks,
                validation_data=self.task.valid_dataset,
                verbose=2
                )
