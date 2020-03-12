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
    tf.print("Epoch %d: " %epoch)
    tf.print("Saved checkpoint for:", save_path)

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, summary_writer):
    self.summary_writer = summary_writer

  def on_train_batch_end(self, batch, logs=None):
    if (self.model.optimizer.iterations % 200) == 0:
      print(logs)
      print(batch.shape)
      if 'loss' in logs.keys():
        log_summary(log_name='learning_rate', log_value=self.model.optimizer.learning_rate( self.model.optimizer.iterations), summary_scope='train')
        log_summary(log_name='fine_total_loss', log_value=logs['loss'], summary_scope='train')
      if 'masked_sequence_loss' in logs.keys():
        log_summary(log_name='fine_lm_loss', log_value=logs['masked_sequence_loss'], summary_scope='train')
      if 'sequence_loss' in logs.keys():
        log_summary(log_name='fine_lm_loss', log_value=logs['sequence_loss'], summary_scope='train')



  def on_epoch_end(self, epoch, logs=None):
    # Log summary for test and train
    if 'masked_sequence_loss' in logs.keys():
      log_summary(log_name='perolexity', log_value=tf.exp(logs['masked_sequence_loss']), summary_scope='train')
      log_summary(log_name='perplexity', log_value=tf.exp(logs['val_masked_sequence_loss']), summary_scope='valid')

    for key in logs.keys():
      if 'val' in key:
        log_summary(log_name=key, log_value=logs[key], summary_scope='valid')
      else:
        log_summary(log_name=key, log_value=logs[key], summary_scope='train')
