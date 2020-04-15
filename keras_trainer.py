import os
import tensorflow as tf
from util import constants
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.trainer import Trainer
from absl import app
from absl import flags

from util.models import MODELS
from util.tasks import TASKS

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'trial1', 'experiment directory')
flags.DEFINE_string('task', 'word_sv_agreement_lm', 'sv_agreement_lm | word_sv_agreement_lm')
flags.DEFINE_string('model', 'lm_lstm', 'lm_lstm | lm_gpt2 | lm_gpt2_shared | lm_lstm_shared_emb | cl_gpt2_shared | cl_gpt2 | cl_lstm')
flags.DEFINE_string('model_config', 'base', 'base | small_lstm ')
flags.DEFINE_string('train_config', 'radam_fast', 'radam_slow | radam_fast')
flags.DEFINE_integer('keep_checkpoint_every_n_hours',None, 'keep_checkpoint_every_n_hours passed to training manager')
flags.DEFINE_integer('batch_size', 64, 'batch_size')

hparams = flags.FLAGS


def run():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Currently, memory growth needs to be the same across GPUs
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)


  log_dir = "logs"
  chkpt_dir = "tf_ckpts"

  strategy = tf.distribute.MirroredStrategy()

  # Create task
  with strategy.scope():
    task = TASKS[hparams.task](get_task_params(batch_size=hparams.batch_size,
                                               num_replicas_in_sync=strategy.num_replicas_in_sync))

    # Create the Model
    model_params = get_model_params(task,hparams.model, hparams.model_config)
    print("model_params: ", model_params.__dict__)

    cl_token = task.sentence_encoder().encode(constants.bos)
    model = MODELS[hparams.model](hparams=get_model_params(task,hparams.model, hparams.model_config),cl_token=cl_token)

  trainer_params = get_train_params(hparams.train_config)

  log_dir = os.path.join(log_dir,task.name, model.model_name+"_"+str(hparams.model_config)+"_"+str(trainer_params.learning_rate)+"_"+hparams.exp_name)
  ckpt_dir = os.path.join(chkpt_dir,task.name, model.model_name+"_"+str(hparams.model_config)+"_"+str(trainer_params.learning_rate)+"_"+hparams.exp_name)

  trainer = Trainer(hparams,
                    strategy=strategy,
                    task=task,
                    model=model,
                    train_params=trainer_params,
                    log_dir=log_dir,
                    ckpt_dir=ckpt_dir)

  trainer.restore()
  trainer.train()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run()

if __name__ == '__main__':
  app.run(main)