class ModelConfig(object):
  def __init__(self,
               hidden_dim=1024,
               embedding_dim=512,
               input_dim=None,
               output_dim=None,
               depth=3,
               hidden_dropout_rate=0.5,
               input_dropout_rate=0.2,
               initializer_range=0.02):
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.depth = depth
    self.hidden_dropout_rate = hidden_dropout_rate
    self.input_dropout_rate = input_dropout_rate
    self.initializer_range = initializer_range

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
        resid_pdrop=0.1,
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


small_lstm = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.25,
  'input_dropout_rate': 0.2,
}

big_lstm = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.25,
  'input_dropout_rate': 0.2,
}


lstm_simple = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.0,
  'input_dropout_rate': 0.0,
}

lstm_drop1 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.1,
  'input_dropout_rate': 0.1,
}

lstm_drop2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.2,
}

MODEL_CONFIGS = {
  'base':{},
  'small_lstm':small_lstm,
  'big_lstm':big_lstm,
  'lstm_simple': lstm_simple,
  'lstm_drop1': lstm_drop1,
  'lstm_drop2': lstm_drop2
}