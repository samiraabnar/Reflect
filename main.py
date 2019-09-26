import os
from absl import app
from absl import flags
from absl import logging

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow.compat.v2.summary as summary

import tqdm

from pipeline.tasks import SvAgreementLM
from task_scripts import get_model_params, get_task_params
from tf2_models.lm_lstm import LmLSTM

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_dir', 'logs', 'experiment directory')


def train_step(examples, model, optimizer, loss_fn):
  """Performs one update step."""
  x, y = examples
  inputs_mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
  trainable_variables = model.trainable_variables
  with tf.GradientTape() as tape:
    logits = model(x, train=True)
    # log_prob_x per datapoint:
    loss = loss_fn(logits, y, inputs_mask)

  grads = tape.gradient(loss, trainable_variables)

  # checking for None valued gradients
  none_grads = np.array([g is None for g in grads])
  if np.any(none_grads):
    none_grads_vars = np.array(trainable_variables)[none_grads]
    message = f'WARNING: None gradients for variables: '
    for var in none_grads_vars:
      message += f'{var.name} '
    logging.warning(message)

  optimizer.apply_gradients(zip(grads, trainable_variables))
  return loss.numpy()


def run(log_dir):
  """Runs experiments based on hyperparameters and log_dir."""

  logging.info('Enabling TF2.')
  tf.enable_v2_behavior()

  num_epochs = 5
  summary_freq = 200
  checkpoint_freq = 1000

  logging.info('Setup: data reader .')
  task = SvAgreementLM(get_task_params())


  logging.info('Setup: summary writer.')
  summary_dir = os.path.join(log_dir, 'summaries')
  tf.io.gfile.makedirs(log_dir)
  train_sw = summary.create_file_writer(os.path.join(summary_dir, 'train'))

  model = LmLSTM(hparams=get_model_params(task))
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  # Checkpointing.
  ckpt = tf.train.Checkpoint(opt=opt)
  manager = tf.train.CheckpointManager(ckpt,
                                       log_dir,
                                       max_to_keep=5,
                                       keep_checkpoint_every_n_hours=6)

  # Force checkpointing of the initial model.
  last_ckpt_step = -10 ** 6
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_step = opt.iterations.numpy() + 1

  @tf.function
  def log_summary(loss, start_ts, start_iter):
    """Produce scalar summaries."""

    with summary.experimental.summary_scope('train'):
      summary.scalar('training_loss', loss)
    # Update training speed timings.
    end_ts = tf.timestamp()
    n_iter = opt.iterations - start_iter
    steps_per_sec = tf.cast(n_iter, tf.float64) / (end_ts - start_ts)
    with summary.experimental.summary_scope('stats'):
      summary.scalar('steps_per_sec', steps_per_sec)
    return end_ts, opt.iterations

  batch_iterator = iter(task.train_dataset)

  first_batch = next(batch_iterator)
  # build models by having first unused call.
  _ = model(first_batch[0])
  # make iterator again so that first data batch is in iterator.
  batch_iterator = iter(task.train_dataset)

  epoch = opt.iterations.numpy() // task.n_train_batches

  num_steps = task.n_train_batches * num_epochs
  summary.experimental.set_step(opt.iterations)
  with train_sw.as_default():
    start_ts, start_iter = tf.timestamp(), opt.iterations.numpy()
    with tqdm.tqdm(total=num_steps) as pbar:
      while True:
        step = opt.iterations.numpy() + 1
        if (step - 1) % task.n_train_batches == 0:
          epoch += 1
          logging.info(f'\nEpoch {epoch: 3d}')
        is_last = step >= num_steps
        is_summary = (is_last or (step % summary_freq == 0)
                      if summary_freq else False)
        is_ckpt = (is_last or (step - last_ckpt_step >= checkpoint_freq)
                   if checkpoint_freq else False)

        batch = next(batch_iterator)
        loss = train_step(batch, model, opt, task.get_loss_fn())

        if is_summary:  # Log summaries.
          message = (f'Iter {step:03d}\t'
                     f'loss = {loss: 7.2f}\t')

          logging.info(message)
          start_ts, start_iter = log_summary(loss, start_ts, start_iter)

        if is_ckpt:  # Checkpoint the model.
          logging.info('Writing checkpoint at step=%d', step)
          manager.save()
          last_ckpt_step = step

        pbar.update(step - pbar.n)
        if is_last:  # Finished training?
          break

    summary.flush(train_sw)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run(FLAGS.experiment_dir)

if __name__ == '__main__':
  app.run(main)