import tensorflow as tf

from tf2_models.utils import log_summary


class CheckpointCallback(tf.keras.callbacks.Callback):

  def __init__(self, manager, ckpt):
    super(CheckpointCallback, self).__init__()
    self.manager = manager
    self.ckpt = ckpt

  def on_epoch_end(self, epoch, logs=None):
    self.ckpt.step.assign_add(1)
    save_path = self.manager.save()
    print("Epoch %d: " %epoch)
    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, summary_writer):
    self.summary_writer = summary_writer

  def on_train_batch_end(self, batch, logs=None):
    if (self.model.optimizer.iterations % 200) == 0:
      # Log LR
      log_summary(log_name='learning_rate', log_value=self.model.optimizer.learning_rate( self.model.optimizer.iterations), summary_scope='train')
      log_summary(log_name='fine_total_loss', log_value=logs['loss'], summary_scope='train')
      log_summary(log_name='fine_lm_loss', log_value=logs['masked_sequence_loss'], summary_scope='train')



  def on_epoch_end(self, epoch, logs=None):
    print(logs)
    # Log summary for test and train
    log_summary(log_name='loss', log_value=logs['masked_sequence_loss'], summary_scope='train')
    log_summary(log_name='perolexity', log_value=tf.exp(logs['masked_sequence_loss']), summary_scope='train')
    log_summary(log_name='loss', log_value=logs['val_masked_sequence_loss'], summary_scope='valid')
    log_summary(log_name='perplexity', log_value=tf.exp(logs['val_masked_sequence_loss']), summary_scope='valid')