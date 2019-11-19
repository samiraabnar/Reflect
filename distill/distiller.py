import tensorflow as tf
import os
from tf2_models.train_utils import ExponentialDecayWithWarmpUp
from tf2_models.trainer import OPTIMIZER_DIC
from tf2_models.utils import camel2snake
from inspect import isfunction
from absl import logging
import numpy as np
class Distiller(object):
  ''' Pipeline for offline distillation.
  '''
  def __init__(self, distill_params, teacher_model, student_model, task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.task = task
    self.distill_params = distill_params
    self.temperature = tf.convert_to_tensor(distill_params.distill_temp)

    self.distill_loss = self.task.get_distill_loss_fn(self.distill_params)
    self.task_loss = self.task.get_loss_fn()
    self.metrics = self.task.metrics()
    self.task_probs_fn = self.task.get_probs_fn()

    self.create_student_optimizer()
    self.setup_ckp_and_summary(student_ckpt_dir, student_log_dir, teacher_ckpt_dir, teacher_log_dir)
    self.setup_models(distill_params, task)

  def create_student_optimizer(self):
    student_initial_learning_rate = self.distill_params.student_learning_rate

    if self.distill_params.schedule == 'cosine_restart':
      lr_schedule = (
        tf.keras.experimental.CosineDecayRestarts(
          student_initial_learning_rate,
          self.distill_params.student_decay_steps))
    else:
      lr_schedule = ExponentialDecayWithWarmpUp(
        initial_learning_rate=student_initial_learning_rate,
        decay_steps=self.distill_params.student_decay_steps,
        decay_rate=0.96,
        warmup_steps=self.distill_params.student_warmup_steps,
        hold_base_rate_steps=self.distill_params.student_hold_base_rate_steps)
    self.student_optimizer = OPTIMIZER_DIC[self.distill_params.student_optimizer](
      learning_rate=lr_schedule, epsilon=1e-08, clipnorm=1.0)

  def setup_ckp_and_summary(self, student_ckpt_dir, student_log_dir, teacher_ckpt_dir, teacher_log_dir):

    # Init checkpoints
    self.teacher_ckpt = tf.train.Checkpoint(net=self.teacher_model)
    self.teacher_manager = tf.train.CheckpointManager(self.teacher_ckpt, teacher_ckpt_dir, max_to_keep=2)
    self.student_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.student_optimizer,
                                            net=self.student_model)
    self.student_manager = tf.train.CheckpointManager(self.student_ckpt, student_ckpt_dir, max_to_keep=2)

    # Init summary
    student_summary_dir = os.path.join(student_log_dir, 'summaries')
    tf.io.gfile.makedirs(student_log_dir)
    self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(student_summary_dir, 'train'))
    tf.compat.v2.summary.experimental.set_step(self.student_optimizer.iterations)

  def setup_models(self, distill_params, task):
    x, y = iter(self.task.valid_dataset).next()
    self.student_model(x, training=True)
    self.student_model.summary()
    self.teacher_model(x, training=True)
    self.teacher_model.summary()
    self.student_model.compile(
      optimizer=self.student_optimizer,
      loss=self.task_loss,
      metrics=[self.metrics])
    self.teacher_model.compile(
      loss=self.task_loss,
      metrics=[self.metrics])

  def restore_teacher(self):
    ''' Restore the teacher model from its checkpoint.
    '''
    self.teacher_ckpt.restore(self.teacher_manager.latest_checkpoint)
    if self.teacher_manager.latest_checkpoint:
      print("Restored teacher from {}".format(self.teacher_manager.latest_checkpoint))
    else:
      print("Initializing teacher from scratch.")

  def restore_student(self):
    ''' Restore the student model from its checkpoint.
    '''
    self.student_ckpt.restore(self.student_manager.latest_checkpoint)
    if self.student_manager.latest_checkpoint:
      print("Restored student from {}".format(self.student_manager.latest_checkpoint))
    else:
      print("Initializing student from scratch.")

  def save_student(self):
    self.student_ckpt.step.assign_add(1)
    save_path = self.student_manager.save()
    tf.print("Saved student checkpoint for step {}: {}".format(int(self.student_ckpt.step), save_path))

  def distill_loop(self):
    ''' Offline Distillation main loop.
    '''
    # logging.info('Distribute strategy: mirrored.')
    # strategy = tf.distribute.MirroredStrategy()
    # train_dataset = strategy.experimental_distribute_dataset(self.task.train_dataset)
    # valid_dataset = strategy.experimental_distribute_dataset(self.task.valid_dataset)

    @tf.function(experimental_relax_shapes=True)
    def student_train_step(x, teacher_y, y_true):
      ''' Training step for the student model (this is the only training step for offline distillation).

      :param x: input
      :param y: output of the teacher model, used to compute distill loss
      :param y_true: actual outputs, used to compute actual loss
      :return:
      distill_loss
      actual_loss
      '''
      scale_distill_grads = np.math.pow(self.distill_params.distill_temp, 2)

      with tf.GradientTape() as tape:
        logits = self.student_model(x, training=True)
        distill_loss = self.distill_loss(y_pred=logits, y_true=teacher_y)
        reg_loss = tf.math.add_n(self.student_model.losses)
        actual_loss = self.task_loss(y_pred=logits, y_true=y_true)
        final_loss = scale_distill_grads * self.distill_params.student_distill_rate * distill_loss + \
                    self.distill_params.student_gold_rate * actual_loss + reg_loss

      grads = tape.gradient(final_loss, self.student_model.trainable_weights)
      self.student_model.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights),
                                                   name="student_optimizer")

      return distill_loss, actual_loss

    @tf.function
    def epoch_loop():
      step = 0
      for x,y in self.task.train_dataset:
        teacher_logits = self.teacher_model(x, training=True)
        teacher_probs = self.task_probs_fn(logits=teacher_logits, labels=y, temperature=self.temperature)
        distill_loss, actual_loss = student_train_step(x=x, teacher_y=teacher_probs, y_true=y)

        # Log every 200 batches.
        if step % 200 == 0:
          with tf.summary.experimental.summary_scope("student_train"):
            tf.summary.scalar('student_learning_rate',
                        self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations),
                        )
            tf.summary.scalar('fine_distill_loss',
                              distill_loss)

        step += 1
        # Stop at the end of the epoch
        if (step % self.task.n_train_batches) == 0:
          with tf.summary.experimental.summary_scope("student_train"):
            tf.summary.scalar('distill_loss', distill_loss)
            tf.summary.scalar('actual_loss', actual_loss)
          break

    @tf.function
    def summarize(teacher_eval_results, student_eval_results):
      with tf.summary.experimental.summary_scope("eval_teacher"):
        for i, m_name in enumerate(self.teacher_model.metrics_names):
          tf.summary.scalar(m_name, teacher_eval_results[i])


      with tf.summary.experimental.summary_scope("eval_student"):
        for i, m_name in enumerate(self.student_model.metrics_names):
          tf.summary.scalar(m_name, student_eval_results[i])

    with self.summary_writer.as_default():
      for _ in np.arange(self.distill_params.n_epochs):
        epoch_loop()
        # Evaluate Teacher
        teacher_eval_results = self.teacher_model.evaluate(self.task.valid_dataset,
                                                           steps=self.task.n_valid_batches)
        # Evaluate Student
        student_eval_results = self.student_model.evaluate(self.task.valid_dataset,
                                                           steps=self.task.n_valid_batches)
        summarize(teacher_eval_results, student_eval_results)

        self.save_student()



