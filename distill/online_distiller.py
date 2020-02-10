import tensorflow as tf
import os
from distill.distiller import Distiller
from tf2_models.train_utils import ExponentialDecayWithWarmpUp
from tf2_models.trainer import OPTIMIZER_DIC
from tf2_models.utils import camel2snake
from inspect import isfunction
import numpy as np

class OnlineDistiller(Distiller):
  def __init__(self, hparams, distill_params, teacher_model, student_model, task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir):
    self.hparams = hparams
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
    self.create_teacher_optimizer()

    self.setup_ckp_and_summary(student_ckpt_dir, student_log_dir, teacher_ckpt_dir, teacher_log_dir)
    self.setup_models(distill_params, task)
    self.setup_loggings()


  def setup_ckp_and_summary(self, student_ckpt_dir, student_log_dir, teacher_ckpt_dir, teacher_log_dir):

    # Init checkpoints
    self.teacher_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.teacher_optimizer,
                                            net=self.teacher_model)
    self.teacher_manager = tf.train.CheckpointManager(self.teacher_ckpt, teacher_ckpt_dir, max_to_keep=2)
    self.student_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.student_optimizer,
                                            net=self.student_model)
    self.student_manager = tf.train.CheckpointManager(self.student_ckpt, student_ckpt_dir, max_to_keep=2)

    # Init summary
    student_summary_dir = os.path.join(student_log_dir, 'summaries')
    tf.io.gfile.makedirs(student_log_dir)
    self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(student_summary_dir, 'train'))
    tf.compat.v2.summary.experimental.set_step(self.teacher_optimizer.iterations)


  def create_teacher_optimizer(self):
    teacher_initial_learning_rate = self.distill_params.teacher_learning_rate
    lr_schedule = ExponentialDecayWithWarmpUp(
      initial_learning_rate=teacher_initial_learning_rate,
      decay_steps=self.distill_params.teacher_decay_steps,
      decay_rate=0.96,
      warmup_steps=self.distill_params.teacher_warmup_steps,
      hold_base_rate_steps=self.distill_params.teacher_hold_base_rate_steps)
    self.teacher_optimizer = OPTIMIZER_DIC[self.distill_params.teacher_optimizer](
      learning_rate=lr_schedule, epsilon=1e-08, clipnorm=1.0)

  def setup_loggings(self):
    self.student_validation_metrics = {}
    for metric in self.metrics:
      if isfunction(metric):
        self.student_validation_metrics[camel2snake(metric.__name__)] = tf.keras.metrics.Mean()
      else:
        self.student_validation_metrics[camel2snake(metric.__class__.__name__)] = tf.keras.metrics.Mean()
    self.student_validation_loss = tf.keras.metrics.Mean()

    self.teacher_validation_metrics = {}
    for metric in self.metrics:
      if isfunction(metric):
        self.teacher_validation_metrics[camel2snake(metric.__name__)] = tf.keras.metrics.Mean()
      else:
        self.teacher_validation_metrics[camel2snake(metric.__class__.__name__)] = tf.keras.metrics.Mean()

  def setup_models(self, distill_params, task):
    x, y = iter(self.task.valid_dataset).next()
    self.student_model(x)
    self.student_model.summary()
    self.teacher_model(x)
    self.teacher_model.summary()

    self.student_model.compile(
      optimizer=self.student_optimizer,
      loss=self.task_loss,
      metrics=[self.metrics])

    self.teacher_model.compile(
      optimizer=self.teacher_optimizer,
      loss=self.task_loss,
      metrics=[self.metrics])

  def distill_loop(self):
    @tf.function(experimental_relax_shapes=True)
    def teacher_train_step(x, y_true):
      with tf.GradientTape() as tape:
        logits = self.teacher_model(x, training=True)
        loss = self.teacher_model.loss(y_pred=logits, y_true=y_true)
        if len(self.teacher_model.losses) > 0:
          reg_loss = tf.math.add_n(self.teacher_model.losses)
        else:
          reg_loss = 0
        final_loss = loss + reg_loss

      grads = tape.gradient(final_loss, self.teacher_model.trainable_weights)
      self.teacher_model.optimizer.apply_gradients(zip(grads, self.teacher_model.trainable_weights),
                                                   name="teacher_optimizer")

      return logits, final_loss

    @tf.function(experimental_relax_shapes=True)
    def student_train_step(x, y, y_true):
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
        distill_loss = self.distill_loss(y_pred=logits, y_true=y)
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
      for x, y in self.task.train_dataset:
        teacher_logits, teacher_loss = teacher_train_step(x, y)
        teacher_probs = self.task_probs_fn(logits=teacher_logits, labels=y, temperature=self.temperature)
        soft_targets = tf.stop_gradient(teacher_probs)
        distill_loss, actual_loss = student_train_step(x=x, y=soft_targets, y_true=y)

        # Log every 200 batches.
        if step % 200 == 0:
          with tf.summary.experimental.summary_scope("student_train"):
            tf.summary.scalar('student_learning_rate',
                          self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations))
            tf.summary.scalar('fine_distill_loss', distill_loss, )
          with tf.summary.experimental.summary_scope("teacher_train"):
            tf.summary.scalar('teacher_loss', teacher_loss)
            tf.summary.scalar('teacher_learning_rate',
                              self.teacher_model.optimizer.learning_rate(self.teacher_model.optimizer.iterations))

        step += 1
        if step == self.task.n_train_batches:
          with tf.summary.experimental.summary_scope("student_train"):
            tf.summary.scalar('distill_loss', distill_loss)
            tf.summary.scalar('actual_loss', actual_loss)
          break



    with self.summary_writer.as_default():
      num_epochs = self.distill_params.n_epochs
      for _ in tf.range(num_epochs):
        epoch_loop()

        teacher_eval_results = self.teacher_model.evaluate(self.task.valid_dataset,
                                                           steps=self.task.n_valid_batches)

        # Evaluate Teacher
        with tf.summary.experimental.summary_scope("eval_teacher"):
          for i, m_name in enumerate(self.teacher_model.metrics_names):
            tf.summary.scalar(m_name, teacher_eval_results[i])

        # Evaluate Student
        student_eval_results = self.student_model.evaluate(self.task.valid_dataset,
                                                           steps=self.task.n_valid_batches)
        with tf.summary.experimental.summary_scope("eval_student"):
          for i, m_name in enumerate(self.student_model.metrics_names):
            tf.summary.scalar(m_name, student_eval_results[i])

        self.save_student()
        self.save_teacher()

  def save_teacher(self):
    self.teacher_ckpt.step.assign_add(1)
    save_path = self.teacher_manager.save()
    tf.print("Saved teacher checkpoint for step {}: {}".format(int(self.teacher_ckpt.step), save_path))


