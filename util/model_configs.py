class ModelConfig(object):
  def __init__(self,
               hidden_dim=1024,
               embedding_dim=512,
               input_dim=None,
               output_dim=None,
               depth=3,
               hidden_dropout_rate=0.5,
               input_dropout_rate=0.2,
               initializer_range=None):
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


small_gpt = {
  'embedding_dim': 128
}

big_gpt = {
  'embedding_dim': 256
}

very_big_gpt = {
  'embedding_dim': 512
}

very_big_gpt_v2 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.2,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.2
}

very_big_gpt_v3 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.3,
  'attn_pdrop': 0.3
}

very_big_gpt_v4 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.3,
  'attn_pdrop': 0.4
}

very_big_gpt_v5 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.5,
  'attn_pdrop': 0.4
}

very_big_gpt_v6 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.5,
  'embd_pdrop': 0.5,
  'attn_pdrop': 0.5
}

very_big_gpt_v7 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.5,
  'attn_pdrop': 0.4
}

small_gpt_v2 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.1,
  'embd_pdrop': 0.0,
  'attn_pdrop': 0.1
}

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

big_lstm_v2 = {
  'hidden_dim': 512,
  'embedding_dim': 128,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.2,
}

bigger_lstm_v2 = {
  'hidden_dim': 728,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.2,
}

bigger_lstm_v4 = {
  'hidden_dim': 728,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.2,
}

bigger_lstm_v3 = {
  'hidden_dim': 728,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.4,
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

lstm_drop12 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.1,
  'input_dropout_rate': 0.2,
}

lstm_drop3 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.3,
}

lstm_drop30 = {
  'hidden_dim': 512,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.0,
}

lstm_drop20_v2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

lstm3_drop20 = {
  'hidden_dim': 128,
  'embedding_dim': 256,
  'depth': 3,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

lstm3_big_drop2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 3,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.2,
  'initializer_range': 0.1
}

big_lstm_drop5 = {
  'hidden_dim': 1024,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.2,
}

lstm2_big_drop20 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

MODEL_CONFIGS = {
  'base':{},
  'small_lstm':small_lstm,
  'big_lstm':big_lstm,
  'lstm_simple': lstm_simple,
  'lstm_drop1': lstm_drop1,
  'lstm_drop2': lstm_drop2,
  'lstm_drop12': lstm_drop12,
  'lstm_drop3': lstm_drop3,
  'big_lstm_v2': big_lstm_v2,
  'bigger_lstm_v2': bigger_lstm_v2,
  'bigger_lstm_v3': bigger_lstm_v3,
  'bigger_lstm_v4': bigger_lstm_v4,
  'big_lstm_drop5': big_lstm_drop5,
  'lstm_drop30': lstm_drop30,
  'small_gpt': small_gpt,
  'big_gpt': big_gpt,
  'very_big_gpt': very_big_gpt,
  'lstm_drop20_v2': lstm_drop20_v2,
  'lstm3_drop20': lstm3_drop20,
  'very_big_gpt_v2': very_big_gpt_v2,
  'small_gpt_v2': small_gpt_v2,
  'very_big_gpt_v3': very_big_gpt_v3,
  'lstm3_big_drop2': lstm3_big_drop2,
  'very_big_gpt_v4': very_big_gpt_v4,
  'very_big_gpt_v5': very_big_gpt_v5,
  'very_big_gpt_v6': very_big_gpt_v6,
  'lstm2_big_drop20': lstm2_big_drop20,
  'very_big_gpt_v7': very_big_gpt_v7
}