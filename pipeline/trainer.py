import tensorflow as tf
import os
from pipeline import  train_utils


class Trainer(object):
  def __init__(self, model, task, train_params):
    self.n_epochs = 10
    self.model = model
    self.task = task
    self.optimizer =  tf.optimizers.Adam(learning_rate=0.001,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-07,
                                        amsgrad=False,
                                        name='Adam')

  def train(self):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join('tf_ckpts', self.task.name, self.model.model_name),
                                         max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")

    train_summary_writer = tf.summary.create_file_writer('logs/sv_agreement/lm_lstm/train')
    eval_summary_writer = tf.summary.create_file_writer('logs/sv_agreement/lm_lstm/eval')

    train_avg_metric_dic = {}
    for metric in self.task.metrics:
      train_avg_metric_dic[metric] = tf.keras.metrics.Mean(name=metric, dtype=tf.float32)

    eval_avg_metric_dic = {}
    for metric in self.task.metrics:
      eval_avg_metric_dic[metric] = tf.keras.metrics.Mean(name=metric, dtype=tf.float32)


    for i in range(self.n_epochs):
      with train_summary_writer.as_default():
        t_loss = train_utils.train(self.model, self.task.train_dataset,
                       self.optimizer, self.task.get_loss_fn(),
                       metric_dic=train_avg_metric_dic, task=self.task)
      with eval_summary_writer.as_default():
        train_utils.eval(self.model,
             self.task.valid_dataset, self.task.get_loss_fn(),
             metric_dic=eval_avg_metric_dic, task=self.task, step_num=self.optimizer.iterations)

      ckpt.step.assign_add(1)
      if int(ckpt.step) % 10 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("loss@epoch%d:" % i, t_loss)



