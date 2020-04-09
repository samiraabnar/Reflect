from tasks.lm1b import Lm1B
from tasks.mnist import Mnist, AffNistTask
from tasks.smallnorb import SmallNorb
from tasks.sst import ClassifySST2, LmSST2
from tasks.sv_agreement import SvAgreementLM, WordSvAgreementLM, WordSvAgreementVP
from tasks.wiki import WikiLM

TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
  'word_sv_agreement_vp': WordSvAgreementVP,
  'mnist': Mnist,
  'affnist': AffNistTask,
  'smallnorb': SmallNorb,
  'sst2': ClassifySST2,
  'lm_sst2': LmSST2,
  'lm1b': Lm1B,
  'wikilm': WikiLM
}