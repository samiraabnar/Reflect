import tensorflow as tf
import numpy as np
import os
from pipeline.tasks import SvAgreementLM
from pipeline.trainer import Trainer
from tf2_models.lm_lstm import LmLSTM


class TrainParams:
  learning_rate = 0.001
  n_epochs = 30
  warmpup_steps = 10000
  hold_base_rate_steps = 1000
  total_training_steps = 60000


class TaskParams:
  batch_size = 128


class ModelParams:
  hidden_dim=256
  input_dim=None
  output_dim=None
  depth=2
  hidden_dropout_rate=0.5
  input_dropout_rate=0.2


def get_train_params():
  train_params = TrainParams()
  return train_params

def get_task_params():
  task_params = TaskParams()
  return task_params

def get_model_params(task):
  model_params = ModelParams()
  model_params.input_dim = task.databuilder.vocab_size()
  model_params.output_dim = task.databuilder.vocab_size()
  return model_params

if __name__ == '__main__':
  # tf.debugging.set_log_device_placement(True)


  # Create the Task
  task = SvAgreementLM(get_task_params())
  for x,y in task.train_dataset:
    print("Data shape:", x.shape)

  # Create the Model
  model = LmLSTM(hparams=get_model_params(task))



  # Instantiate an optimizer to train the model.
  optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
  # Instantiate a loss function.
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  # Prepare the metrics.
  train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

  # Iterate over epochs.
  for epoch in range(3):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(task.train_dataset):
      with tf.GradientTape() as tape:
        logits = model(x_batch_train)
        loss_value = loss_fn(y_batch_train, logits)
      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Update training metric.
      train_acc_metric(y_batch_train, logits)

      # Log every 200 batches.
      if step % 200 == 0:
        print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
        print('Seen so far: %s samples' % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in task.valid_dataset:
      val_logits = model(x_batch_val)
      # Update val metrics
      val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: %s' % (float(val_acc),))

  #history = model.fit(task.train_dataset, epochs=5)

  # # Create the Trainer
  # trainer = Trainer(model=model, task=task, train_params=get_train_params())
  #
  # #Train
  # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, net=trainer.model)
  # chpt_path = os.path.join('tf_ckpts', trainer.task.name, trainer.model.model_name)
  # manager = tf.train.CheckpointManager(ckpt,
  #                                      chpt_path,
  #                                      max_to_keep=3)
  # ckpt.restore(manager.latest_checkpoint)
  # if manager.latest_checkpoint:
  #   print("Restored from {}".format(manager.latest_checkpoint))
  # else:
  #   print("Initializing from scratch.")
  #   print("Saving params")
  #   if not os.path.exists(chpt_path):
  #     os.makedirs(chpt_path)
  #   np.save(os.path.join(chpt_path, 'task_params'), task.task_params)
  #   np.save(os.path.join(chpt_path, 'model_params'), model.hparams)
  #   np.save(os.path.join(chpt_path, 'train_params'), trainer.train_params)
  #
  #   trainer.train(ckpt, manager)