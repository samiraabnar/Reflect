import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from pipeline.tasks import SvAgreementLM
from task_scripts import get_model_params, get_task_params
from tf2_models.lm_lstm import LmLSTM


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



if __name__ == '__main__':
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

  tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/test',
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq='epoch')  # How often to write logs (default: once per epoch)

  ckpt_callback = CheckpointCallback(manager=manager, ckpt=ckpt)

  model.fit(task.train_dataset,
            epochs=3,
            callbacks=[tb_callback, ckpt_callback])