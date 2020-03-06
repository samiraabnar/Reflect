import tensorflow as tf
import numpy as np
from distill.distill_util import get_distill_scheduler
from distill.distiller import Distiller



class ScheduledDistiller(Distiller):
  ''' Pipeline for offline scheduled distillation.
  '''
  def __init__(self, hparams, distill_params, teacher_model, student_model, task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir):
    super(ScheduledDistiller, self).__init__(hparams, distill_params, teacher_model, student_model, task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir)

    self.distillrate_scheduler = get_distill_scheduler(distill_params.distill_schedule,
                                                       min=distill_params.distill_min_rate,
                                                       max=distill_params.student_distill_rate)

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
      student_distill_rate = self.distillrate_scheduler(self.student_optimizer.iterations)
      student_gold_rate = 1 - student_distill_rate
      with tf.GradientTape() as tape:
        logits = self.student_model(x, training=True)
        distill_loss = self.distill_loss(y_pred=logits, y_true=teacher_y)
        reg_loss = tf.math.add_n(self.student_model.losses)
        actual_loss = self.task_loss(y_pred=logits, y_true=y_true)
        final_loss = scale_distill_grads * student_distill_rate * distill_loss + \
                    student_gold_rate * actual_loss + reg_loss

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



