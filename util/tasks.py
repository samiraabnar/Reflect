from tasks.mnist import Mnist, AffNistTask
from tasks.sst import SST2
from tasks.sv_agreement import SvAgreementLM, WordSvAgreementLM, WordSvAgreementVP

TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
  'word_sv_agreement_vp': WordSvAgreementVP,
  'mnist': Mnist,
  'affnist': AffNistTask,
  'sst2': SST2,
}