from tf2_models.lm_lstm import LmLSTM, LmLSTMSharedEmb, ClassifierLSTM
from tf2_models.lm_transformer import LmGPT2, LmGPT2SharedWeights, ClassifierGPT2

MODELS = {"lm_lstm": LmLSTM,
          "lm_gpt2": LmGPT2,
          "lm_gpt2_shared": LmGPT2SharedWeights,
          "lm_lstm_shared_emb": LmLSTMSharedEmb,
          'cl_gpt2': ClassifierGPT2,
          'cl_lstm': ClassifierLSTM}