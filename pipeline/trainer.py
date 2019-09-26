import tensorflow as tf
import os
import numpy as np
from pipeline import  train_utils
from pipeline.train_utils import cosine_decay_with_warmup


class Trainer(object):
  def __init__(self, model, task, train_params):
    self.n_epochs = train_params.n_epochs
    self.learning_rate_base = train_params.learning_rate
    self.total_training_steps = train_params.total_training_steps
    self.train_params = train_params
    self.model = model
    self.task = task

    with tf.device('/gpu:0'):
      self.learning_rate = cosine_decay_with_warmup(tf.compat.v1.train.get_or_create_global_step(),
                                                    self.learning_rate_base,
                                                    total_steps=self.total_training_steps,
                                                    warmup_learning_rate=0.0,
                                                    warmup_steps=train_params.warmpup_steps,
                                                    hold_base_rate_steps=train_params.hold_base_rate_steps)

      self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate,
                                          beta_1=0.9,
                                          beta_2=0.999,
                                          epsilon=1e-07,
                                          amsgrad=False,
                                          name='Adam')
      self.optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

  def train(self):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
    chpt_path = os.path.join('tf_ckpts', self.task.name, self.model.model_name)
    manager = tf.train.CheckpointManager(ckpt,
                                         chpt_path,
                                         max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")
      print("Saving params")
      if not os.path.exists(chpt_path):
        os.makedirs(chpt_path)
      np.save(os.path.join(chpt_path, 'task_params'), self.task.task_params)
      np.save(os.path.join(chpt_path, 'model_params'), self.model.hparams)
      np.save(os.path.join(chpt_path, 'train_params'), self.train_params)

    summary_path = os.path.join('logs', self.task.name, self.model.model_name)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_path, 'train'))
    eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_path, 'eval'))

    train_avg_metric_dic = {}
    for metric in self.task.metrics:
      train_avg_metric_dic[metric] = tf.keras.metrics.Mean(name=metric, dtype=tf.float32)

    eval_avg_metric_dic = {}
    for metric in self.task.metrics:
      print(metric)
      eval_avg_metric_dic[metric] = tf.keras.metrics.Mean(name=metric, dtype=tf.float32)

    print("Start training ... ")
    print("initial learning rate is ", self.optimizer.learning_rate)
    for i in range(self.n_epochs):
      with train_summary_writer.as_default():
        t_loss = train_utils.train(self.model, self.task.train_dataset,
                       self.optimizer, self.task.get_loss_fn(),
                       avg_metric_dic=train_avg_metric_dic, task=self.task)
      with eval_summary_writer.as_default():
        train_utils.eval(self.model,
             self.task.valid_dataset,
             avg_metric_dic=eval_avg_metric_dic, task=self.task, step_num=self.optimizer.iterations)

      ckpt.step.assign_add(1)
      if int(ckpt.step) % 10 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("loss@epoch%d:" % i, t_loss)



