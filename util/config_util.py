class TrainParams:
  learning_rate = 0.001
  n_epochs = 60
  warmpup_steps = 1000
  hold_base_rate_steps = 0
  total_training_steps = 60000
  num_train_epochs = 60
  optimizer = 'adam'

class TaskParams:
  batch_size = 128


class ModelParams:
  hidden_dim=1024
  input_dim=None
  output_dim=None
  depth=2
  hidden_dropout_rate=0.5
  input_dropout_rate=0.2

class GPT2Config(object):
    """Configuration class to store the configuration of a `GPT2Model`.
    Args:
        vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
        n_positions: Number of positional embeddings.
        n_ctx: Size of the causal mask (usually same as n_positions).
        n_embd: Dimensionality of the embeddings and hidden states.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        layer_norm_epsilon: epsilon to use in the layer norm layers
        resid_pdrop: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attn_pdrop: The dropout ratio for the attention
            probabilities.
        embd_pdrop: The dropout ratio for the embeddings.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
    """
    def __init__(
        self,
        vocab_size,
        n_positions=1024,
        n_ctx=1024,
        embedding_dim=512,
        depth=6,
        n_head=8,
        resid_pdrop=0.2,
        embd_pdrop=0.2,
        attn_pdrop=0.2,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        num_labels=1,
        summary_type='cls_index',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs
    ):
      """Constructs GPT2Config.
      Args:
          vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
          n_positions: Number of positional embeddings.
          n_ctx: Size of the causal mask (usually same as n_positions).
          n_embd: Dimensionality of the embeddings and hidden states.
          n_layer: Number of hidden layers in the Transformer encoder.
          n_head: Number of attention heads for each attention layer in
              the Transformer encoder.
          layer_norm_epsilon: epsilon to use in the layer norm layers
          resid_pdrop: The dropout probabilitiy for all fully connected
              layers in the embeddings, encoder, and pooler.
          attn_pdrop: The dropout ratio for the attention
              probabilities.
          embd_pdrop: The dropout ratio for the embeddings.
          initializer_range: The sttdev of the truncated_normal_initializer for
              initializing all weight matrices.
      """
      self.vocab_size = vocab_size
      self.n_ctx = n_ctx
      self.n_positions = n_positions
      self.embedding_dim = embedding_dim
      self.depth = depth
      self.n_head = n_head
      self.resid_pdrop = resid_pdrop
      self.embd_pdrop = embd_pdrop
      self.attn_pdrop = attn_pdrop
      self.layer_norm_epsilon = layer_norm_epsilon
      self.initializer_range = initializer_range

      self.num_labels = num_labels
      self.summary_type = summary_type
      self.summary_use_proj = summary_use_proj
      self.summary_activation = summary_activation
      self.summary_first_dropout = summary_first_dropout
      self.summary_proj_to_labels = summary_proj_to_labels

      self.output_attentions = kwargs.pop('output_attentions', False)
      self.output_hidden_states = kwargs.pop('output_hidden_states', False)

    @property
    def max_position_embeddings(self):
      return self.n_positions

    @property
    def hidden_size(self):
      return self.embedding_dim

    @property
    def num_attention_heads(self):
      return self.n_head

    @property
    def num_hidden_layers(self):
      return self.depth

def get_train_params():
  train_params = TrainParams()
  return train_params

def get_task_params():
  task_params = TaskParams()
  return task_params

def get_model_params(task, config_name=''):
  if config_name == 'lm_gpt2':
    return GPT2Config(vocab_size=task.databuilder.vocab_size())
  else:
    model_params = ModelParams()
    model_params.input_dim = task.databuilder.vocab_size()
    model_params.output_dim = task.databuilder.vocab_size()
    return model_params


