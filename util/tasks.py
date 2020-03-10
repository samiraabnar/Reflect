from tasks.lm1b import Lm1B
from tasks.mnist import Mnist, AffNistTask
from tasks.sst import ClassifySST2
from tasks.sv_agreement import SvAgreementLM, WordSvAgreementLM, WordSvAgreementVP

TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
  'word_sv_agreement_vp': WordSvAgreementVP,
  'mnist': Mnist,
  'affnist': AffNistTask,
  'sst2': ClassifySST2,
  'lm1b': Lm1B
}