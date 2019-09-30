import os
from tasks.tasks import SvAgreementLM
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.lm_lstm import LmLSTM
from tf2_models.trainer import Trainer
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_dir', 'logs', 'experiment directory')

def run():
  log_dir = "logs"
  chkpt_dir = "tf_ckpts"

  task = SvAgreementLM(get_task_params())
  # Create the Model
  model = LmLSTM(hparams=get_model_params(task))

  log_dir = os.path.join(log_dir,task.name, model.model_name)
  ckpt_dir = os.path.join(chkpt_dir,task.name, model.model_name)

  trainer = Trainer(task=task,
                    model=model,
                    train_params=get_train_params(),
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