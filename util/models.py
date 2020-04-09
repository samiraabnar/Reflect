from tf2_models.capnet import Capsule
from tf2_models.cnn import VanillaCNN
from tf2_models.ff import VanillaFF
from tf2_models.lm_lstm import LmLSTM, LmLSTMSharedEmb, ClassifierLSTM, LmLSTMSharedEmbV2
from tf2_models.lm_transformer import LmGPT2, LmGPT2SharedWeights, ClassifierGPT2, ClassifierGPT2SharedWeights, \
  ClassifierBERT, ClassifierBERTSharedWeights
from tf2_models.matrix_caps import MatrixCaps

MODELS = {"lm_lstm": LmLSTM,
          "lm_gpt2": LmGPT2,
          "lm_gpt2_shared": LmGPT2SharedWeights,
          "lm_lstm_shared_emb": LmLSTMSharedEmbV2,
          'cl_gpt2': ClassifierGPT2,
          'cl_lstm': ClassifierLSTM,
          'cl_gpt2_shared': ClassifierGPT2SharedWeights,
          'cl_bert': ClassifierBERT,
          'cl_bert_shared': ClassifierBERTSharedWeights,
          'cl_vcnn': VanillaCNN,
          'cl_vff': VanillaFF,
          'cl_capsule': Capsule,
          'matrix_capsule': MatrixCaps}