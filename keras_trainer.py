import os
import tensorflow as tf
from tasks.tasks import SvAgreementLM, WordSvAgreementLM
from tf2_models.lm_transformer import LmGPT2
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.lm_lstm import LmLSTM, LmLSTMSharedEmb
from tf2_models.trainer import Trainer
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'trial1', 'experiment directory')
flags.DEFINE_string('task', 'word_sv_agreement_lm', 'sv_agreement_lm | word_sv_agreement_lm')
flags.DEFINE_string('model', 'lm_lstm', 'lm_lstm | lm_gpt2 | lm_lstm_shared_emb')
flags.DEFINE_string('model_config', 'base', 'base | small_lstm ')
flags.DEFINE_string('optimizer', 'adam', 'adam | radam ')


hparams = flags.FLAGS


MODELS = {"lm_lstm": LmLSTM,
          "lm_gpt2": LmGPT2,
          "lm_lstm_shared_emb": LmLSTMSharedEmb}

TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
}
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

  # Create task
  task = TASKS[hparams.task](get_task_params())

  # Create the Model
  model_params = get_model_params(task,hparams.model, hparams.model_config)
  print("model_params: ", model_params.__dict__)

  model = MODELS[hparams.model](hparams=get_model_params(task,hparams.model, hparams.model_config))

  trainer_params = get_train_params(hparams.optimizer)

  log_dir = os.path.join(log_dir,task.name, model.model_name+"_"+str(hparams.model_config)+"_"+str(trainer_params.learning_rate)+"_"+hparams.exp_name)
  ckpt_dir = os.path.join(chkpt_dir,task.name, model.model_name+"_"+str(hparams.model_config)+"_"+str(trainer_params.learning_rate)+"_"+hparams.exp_name)

  trainer = Trainer(task=task,
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