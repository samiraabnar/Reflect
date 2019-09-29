import tensorflow as tf
import os
from absl import app
from absl import flags
from absl import logging

from pipeline.tasks import SvAgreementLM
from task_scripts import get_model_params, get_task_params
from tf2_models.lm_lstm import LmLSTM


@tf.function
def log_summary(log_value, log_name, summary_scope):
  """Produce scalar summaries."""
  with tf.compat.v2.summary.experimental.summary_scope(summary_scope):
    tf.compat.v2.summary.scalar(log_name, log_value)



class CheckpointCallback(tf.keras.callbacks.Callback):

  def __init__(self, manager, ckpt):
    super(CheckpointCallback, self).__init__()
    self.manager = manager
    self.ckpt = ckpt

  def on_train_epoch_end(self, epoch, logs=None):
    self.ckpt.step.assign_add(1)
    save_path = self.manager.save()
    print("Epoch %d: " %epoch)
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, summary_writer):
    self.summary_writer = summary_writer

  def on_batch_end(self, batch, logs=None):
    if (self.model.optimizer.iterations % 200) == 0:
      # Log LR
      log_summary(log_name='learning_rate', log_value=tf.keras.backend.eval(self.model.optimizer.lr), summary_scope='train')

  def on_train_epoch_end(self, epoch, logs=None):
    # Log loss
    log_summary(log_name='loss', log_value=logs['loss'], summary_scope='train')


  def on_valid_epoch_end(self, epoch, logs=None):
    # Log loss
    log_summary(log_name='loss', log_value=logs['loss'], summary_scope='valid')


if __name__ == '__main__':
  log_dir = "logs/test"

  task = SvAgreementLM(get_task_params())

  # Create the Model
  model = LmLSTM(hparams=get_model_params(task))

  initial_learning_rate = 0.1
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

  optimizer = tf.keras.optimizers.RMSprop(lr_schedule)

  model.compile(
    optimizer=optimizer,
    loss=tf.losses.sparse_categorical_crossentropy)



  # Checkpointing.
  ckpt = tf.train.Checkpoint(opt=optimizer)
  manager = tf.train.CheckpointManager(ckpt,
                                       'logs/test',
                                       max_to_keep=5,
                                       keep_checkpoint_every_n_hours=6)

  # Force checkpointing of the initial model.
  ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
  manager = tf.train.CheckpointManager(ckpt, 'tf_ckpts', max_to_keep=3)

  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  summary_dir = os.path.join(log_dir, 'summaries')
  tf.io.gfile.makedirs(log_dir)
  summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(summary_dir, 'train'))

  ckpt_callback = CheckpointCallback(manager=manager, ckpt=ckpt)
  summary_callback = SummaryCallback(summary_writer=summary_writer)

  with summary_writer.as_default():
    model.fit(task.train_dataset,
              epochs=3,
              steps_per_epoch=task.n_train_batches,
              validation_steps=task.n_valid_batches,
              callbacks=[ckpt_callback, summary_callback],
              validation_data=task.valid_dataset,
              )