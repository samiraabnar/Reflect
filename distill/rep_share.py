import tensorflow as tf
import os
from distill.distiller import Distiller
from distill.online_distiller import OnlineDistiller
from distill.repsim_util import get_reps
from tf2_models.train_utils import ExponentialDecayWithWarmpUp
from tf2_models.trainer import OPTIMIZER_DIC
from tf2_models.utils import camel2snake
from inspect import isfunction
import numpy as np

class OnlineRepDistiller(OnlineDistiller):
  """
  Implementation of soft representation sharing in online mode
  """
  def __init__(self, hparams, distill_params, strategy, teacher_model, student_model,
               teacher_task, student_task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.strategy = strategy
    self.task = teacher_task
    self.hparams = hparams
    self.distill_params = distill_params
    self.temperature = tf.convert_to_tensor(distill_params.distill_temp)

    self.rep_loss = self.task.get_rep_loss()
    self.student_task_loss = self.task.get_loss_fn()
    self.teacher_task_loss = self.task.get_loss_fn()
    self.student_distill_loss = self.task.get_distill_loss_fn(distill_params=self.distill_params)

    self.student_metrics = self.task.metrics()
    self.teacher_metrics = self.task.metrics()
    self.teacher_task_probs_fn = self.task.get_probs_fn()

    self.train_batch_iterator = iter(self.task.train_dataset)
    #with self.strategy.scope():
    self.create_student_optimizer()
    self.create_teacher_optimizer()

    self.setup_ckp_and_summary(student_ckpt_dir, student_log_dir, teacher_ckpt_dir, teacher_log_dir)
    self.setup_models(distill_params)

  def setup_models(self, distill_params):
    x_s, y_s = iter(self.task.valid_dataset).next()

    self.student_model(x_s)
    self.student_model.summary()
    self.teacher_model(x_s)
    self.teacher_model.summary()

    self.student_model.compile(
      optimizer=self.student_optimizer,
      loss=self.student_task_loss,
      metrics=[self.student_metrics])

    self.teacher_model.compile(
      optimizer=self.teacher_optimizer,
      loss=self.teacher_task_loss,
      metrics=[self.teacher_metrics])


  def distill_loop(self):

    @tf.function(experimental_relax_shapes=True)
    def get_teacher_outputs(x, y_true):
      outputs = self.teacher_model.detailed_call(x, training=tf.convert_to_tensor(True))
      teacher_logits, teacher_reps = outputs[0], outputs[self.teacher_model.rep_index]
      if self.teacher_model.rep_layer != -1 and self.teacher_model.rep_layer is not None:
        teacher_reps = teacher_reps[self.teacher_model.rep_layer]

      loss = self.teacher_model.loss(y_pred=teacher_logits, y_true=y_true)

      return teacher_reps, teacher_logits, loss

    @tf.function(experimental_relax_shapes=True)
    def teacher_train_step(x, y_true):
      with tf.GradientTape() as tape:

        teacher_reps, teacher_logits, loss = get_teacher_outputs(x, y_true)

        if len(self.teacher_model.losses) > 0:
          reg_loss = tf.math.add_n(self.teacher_model.losses)
        else:
          reg_loss = 0
        final_loss = loss + reg_loss

      grads = tape.gradient(final_loss, self.teacher_model.trainable_weights)
      self.teacher_model.optimizer.apply_gradients(zip(grads, self.teacher_model.trainable_weights),
                                                   name="teacher_optimizer")

      return teacher_logits, teacher_reps, final_loss

    @tf.function(experimental_relax_shapes=True)
    def get_student_outputs(x, y_s, teacher_probs, teacher_reps):
      outputs = self.student_model.detailed_call(x, training=tf.convert_to_tensor(True))
      logits, student_reps = outputs[0], outputs[self.student_model.rep_index]
      if self.student_model.rep_layer != -1 and self.student_model.rep_layer is not None:
        student_reps = teacher_reps[self.student_model.rep_layer]

      return student_reps, logits

    @tf.function(experimental_relax_shapes=True)
    def student_train_step(x, y_s, teacher_probs, teacher_reps):
      ''' Training step for the student model (this is the only training step for offline distillation).

      :param x: input
      :param y: output of the teacher model, used to compute distill loss
      :param y_true: actual outputs, used to compute actual loss
      :return:
      distill_loss
      actual_loss
      '''
      with tf.GradientTape() as tape:
        student_reps, logits = get_student_outputs(x, y_s, teacher_probs, teacher_reps)

        rep_loss = self.rep_loss(reps1=student_reps, reps2=teacher_reps,
                                 padding_symbol=tf.constant(self.task.output_padding_symbol))
        actual_loss = self.student_task_loss(y_pred=logits, y_true=y_s)
        distill_loss = self.student_distill_loss(y_pred=logits, y_true=teacher_probs)
        reg_loss = tf.math.add_n(self.student_model.losses)

        final_loss = self.distill_params.student_distill_rep_rate * rep_loss + \
                     self.distill_params.student_gold_rate * actual_loss + \
                     self.distill_params.student_distill_rate * distill_loss + \
                     reg_loss

      grads = tape.gradient(final_loss, self.student_model.trainable_weights)
      self.student_model.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights),
                                                   name="student_optimizer")
      return distill_loss, rep_loss, actual_loss

    @tf.function(experimental_relax_shapes=True)
    def epoch_teacher(x_s, y_s):
      teacher_logits, teacher_reps, teacher_loss = teacher_train_step(x_s, y_s)
      return teacher_logits, teacher_reps, teacher_loss

    @tf.function(experimental_relax_shapes=True)
    def epoch_student(x_s, y_s, teacher_probs, teacher_reps):

      distill_loss, rep_loss, actual_loss = student_train_step(x_s, y_s, teacher_probs, teacher_reps)

      return distill_loss, rep_loss, actual_loss


    @tf.function(experimental_relax_shapes=True)
    def epoch_teacher_probs(teacher_logits, y_s):
      teacher_probs = self.teacher_task_probs_fn(teacher_logits, y_s, tf.constant(self.temperature))

      return teacher_probs

    def epoch_loop(one_epoch_iterator):
      step = 0
      for x_s, y_s in one_epoch_iterator:
        teacher_logits, teacher_reps, teacher_loss = epoch_teacher(x_s, y_s)
        teacher_probs = epoch_teacher_probs(teacher_logits, y_s)
        distill_loss, rep_loss, actual_loss = epoch_student(x_s, y_s, teacher_probs, teacher_reps)

        # Log every 1000 batches.
        if step % 1000 == 0:
          with tf.summary.experimental.summary_scope("student_train"):
            tf.summary.scalar('student_learning_rate',
                              self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations))
            tf.summary.scalar('fine_rep_loss', rep_loss)
            tf.summary.scalar('fine_distill_loss', distill_loss)
            tf.summary.scalar('fine_actual_loss', actual_loss)
            tf.summary.scalar('student_learning_rate',
                              self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations))

          with tf.summary.experimental.summary_scope("teacher_train"):
            tf.summary.scalar('teacher_loss', teacher_loss)
            tf.summary.scalar('teacher_learning_rate',
                              self.teacher_model.optimizer.learning_rate(self.teacher_model.optimizer.iterations))

        if step == self.task.n_train_batches:
          with tf.summary.experimental.summary_scope("student_train"):
            tf.summary.scalar('rep_loss', rep_loss)
            tf.summary.scalar('actual_loss', actual_loss)

        step += 1


    with self.summary_writer.as_default():
      num_epochs = self.distill_params.n_epochs
      for _ in tf.range(num_epochs):
        one_epoch_iterator = (next(self.train_batch_iterator) for _ in range(self.task.n_train_batches))
        epoch_loop(one_epoch_iterator)

        # Evaluate teacher
        teacher_eval_results = self.teacher_model.evaluate(self.task.valid_dataset,
                                                           steps=self.task.n_valid_batches)
        # Evaluate Student
        student_eval_results = self.student_model.evaluate(self.task.valid_dataset,
                                                           steps=self.task.n_valid_batches)

        with tf.summary.experimental.summary_scope("eval_teacher"):
          for i, m_name in enumerate(self.teacher_model.metrics_names):
            tf.summary.scalar(m_name, teacher_eval_results[i])

        with tf.summary.experimental.summary_scope("eval_student"):
          for i, m_name in enumerate(self.student_model.metrics_names):
            tf.summary.scalar(m_name, student_eval_results[i])

        self.save_student()
        self.save_teacher()

