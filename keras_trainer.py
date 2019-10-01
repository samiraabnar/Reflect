import os
from tasks.tasks import SvAgreementLM, WordSvAgreementLM
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.lm_lstm import LmLSTM
from tf2_models.trainer import Trainer
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'trial1', 'experiment directory')
flags.DEFINE_string('task', 'sv_agreement_lm', 'sv_agreement_lm | word_sv_agreement')

hparams = flags.FLAGS




TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
}
def run():
  log_dir = "logs"
  chkpt_dir = "tf_ckpts"

  # Create task
  task = TASKS[hparams.task](get_task_params())

  # Create the Model
  model = LmLSTM(hparams=get_model_params(task))

  log_dir = os.path.join(log_dir,task.name, model.model_name+"_"+hparams.exp_name)
  ckpt_dir = os.path.join(chkpt_dir,task.name, model.model_name+"_"+hparams.exp_name)

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