import tensorflow as tf
from distill.distill_util import DistillLoss
from distill.distiller import Distiller
from util import constants
from util.config_util import get_task_params, get_model_params, get_distill_params
import os
from tasks.tasks import SvAgreementLM, WordSvAgreementLM, WordSvAgreementVP
from tf2_models.lm_transformer import LmGPT2, LmGPT2SharedWeights, ClassifierGPT2
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.lm_lstm import LmLSTM, LmLSTMSharedEmb, ClassifierLSTM
from absl import flags
import sys

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'word_sv_agreement_lm', 'sv_agreement_lm | word_sv_agreement_lm')
flags.DEFINE_string('distill_config', 'base', ' distillation hparams set')

flags.DEFINE_string('teacher_exp_name', 'trial4', 'experiment directory')
flags.DEFINE_string('teacher_model', 'lm_lstm', 'lm_lstm | lm_gpt2')

flags.DEFINE_string('student_exp_name', 'trial1', 'experiment directory')
flags.DEFINE_string('student_model', 'lm_lstm', 'lm_lstm | lm_gpt2')

flags.DEFINE_string('student_config', 'base', 'base | small_lstm ')
flags.DEFINE_string('teacher_config', 'base', 'base | small_lstm ')


FLAGS(sys.argv)
hparams = flags.FLAGS


hparams = flags.FLAGS


MODELS = {"lm_lstm": LmLSTM,
          "lm_gpt2": LmGPT2,
          "lm_gpt2_shared": LmGPT2SharedWeights,
          "lm_lstm_shared_emb": LmLSTMSharedEmb,

          'cl_gpt2': ClassifierGPT2,
          'cl_lstm': ClassifierLSTM}

TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
  'word_sv_agreement_vp': WordSvAgreementVP
}

if __name__ == '__main__':
  log_dir = "logs"
  chkpt_dir = "tf_ckpts"


  # Create task
  task = TASKS[hparams.task](get_task_params())

  # Create the Model
  cl_token = task.databuilder.sentence_encoder().encode(constants.bos)
  teacher_model = MODELS[hparams.teacher_model](hparams=get_model_params(task, hparams.teacher_model, hparams.teacher_config), cl_token=cl_token)
  student_model = MODELS[hparams.student_model](hparams=get_model_params(task, hparams.student_model, hparams.student_config), cl_token=cl_token)

  teacher_log_dir = os.path.join(log_dir, task.name, "teacher_"+teacher_model.model_name + "_" + hparams.teacher_exp_name)
  teacher_ckpt_dir = os.path.join(chkpt_dir, task.name, teacher_model.model_name + "_" + hparams.teacher_exp_name)

  student_log_dir = os.path.join(log_dir, task.name, "student_"+student_model.model_name + "_" + hparams.teacher_exp_name)
  student_ckpt_dir = os.path.join(chkpt_dir, task.name, "student_"+student_model.model_name + "_" + hparams.student_exp_name)



  distiller = Distiller(get_distill_params(hparams.distill_config), teacher_model, student_model, task,
                        teacher_ckpt_dir=teacher_ckpt_dir,
                        teacher_log_dir=teacher_log_dir,
                        student_ckpt_dir=student_ckpt_dir,
                        student_log_dir=student_log_dir,
                        )
  distiller.restore_teacher()
  distiller.restore_student()
  distiller.distill_loop()