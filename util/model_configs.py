class ModelConfig(object):
  def __init__(self,
               hidden_dim=1024,
               embedding_dim=512,
               input_dim=None,
               output_dim=None,
               depth=1,
               hidden_dropout_rate=0.5,
               input_dropout_rate=0.2,
               initializer_range=None,
               filters=[32],
               maxout_size=[32],
               kernel_size=[(3,3)],
               pool_size=[(2,2)],
               proj_depth=1,
               routings=3,
               fc_dim=[],
               **kwargs):
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.depth = depth
    self.proj_depth = proj_depth
    self.fc_dim = fc_dim
    self.hidden_dropout_rate = hidden_dropout_rate
    self.input_dropout_rate = input_dropout_rate
    self.initializer_range = initializer_range
    self.kernel_size = kernel_size
    self.filters = filters
    self.maxout_size = maxout_size
    self.pool_size = pool_size
    self.output_hidden_states = kwargs.pop('output_hidden_states', False)
    self.output_embeddings = kwargs.pop('output_embeddings', False)
    self.routings = routings

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
      self.output_embeddings = kwargs.pop('output_embeddings', False)

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


class CapsConfig(object):
  def __init__(self,
               output_dim=10,
               A=32,
               B=32,
               C=32,
               D=32,
               epsilon=1e-9,
               l2=0.0000002,
               final_lambda=0.01,
               iter_routing=2):
    self.output_dim = output_dim
    self.A = A
    self.B = B
    self.C = C
    self.D = D
    self.epsilon = epsilon
    self.l2 = l2
    self.final_lambda = final_lambda
    self.iter_routing = iter_routing

class ResnetConfig(object):
  def __init__(self, **kwargs):
    self.output_dim =kwargs.get('output_dim', 1)
    self.hidden_dim = kwargs.get('hidden_dim', 512)
    self.pool_size = kwargs.get('pool_size', 3)
    self.filters = kwargs.get('filters', [32, 32, 32, 32])
    self.kernel_size = kwargs.get('kernel_size', [(3, 3), (3, 3), (3, 3), (3, 3)])
    self.hidden_dropout_rate = kwargs.get('hidden_dropout_rate', 0.2)
    self.input_dropout_rate = kwargs.get('input_dropout_rate', 0.0)
    self.num_res_net_blocks = kwargs.get('num_res_net_blocks', 2)


small_gpt = {
  'embedding_dim': 128,
  'resid_pdrop': 0.1,
  'embd_pdrop': 0.1,
  'attn_pdrop': 0.1
}

small_gpt_v3 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.1,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.2
}

small_gpt_v4 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.2,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.2
}

small_gpt_v5 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.3
}

small_gpt_v6 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.5,
  'initializer_range': 0.05
}

small_gpt_v7 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.5,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.5,
  'initializer_range': 0.05
}

small_gpt_v8 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.5,
  'initializer_range': 0.01
}

small_gpt_v9 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.6,
  'initializer_range': 0.05
}


small_ugpt_v9 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.6,
  'initializer_range': 0.05
}

short_gpt_v9 = {
  'embedding_dim': 128,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.6,
  'initializer_range': 0.05,
  'depth': 4
}

big_gpt_v2 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.2,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.2
}

big_gpt_v3 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.3
}

big_gpt_v4 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.3,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.3,
  'initializer_range': 0.05
}

big_gpt_v5 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.2,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.3,
  'initializer_range': 0.05
}

big_gpt_v6 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.2,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.4,
  'initializer_range': 0.05
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
  'embd_pdrop': 0.3,
  'attn_pdrop': 0.4,
  'initializer_range': 0.05
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

very_big_gpt_v8 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.5,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.5
}

very_big_gpt_v9 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.5,
  'initializer_range': 0.05
}

very_big_gpt_v10 = {
  'embedding_dim': 512,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.6,
  'initializer_range': 0.05
}

big_gpt_v9 = {
  'embedding_dim': 256,
  'resid_pdrop': 0.4,
  'embd_pdrop': 0.2,
  'attn_pdrop': 0.5,
  'initializer_range': 0.05
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
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

small_lstm_v2 = {
  'hidden_dim': 256,
  'embedding_dim': 128,
  'depth': 2,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

small_lstm_v3 = {
  'hidden_dim': 256,
  'embedding_dim': 128,
  'depth': 2,
  'hidden_dropout_rate': 0.8,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

small_lstm_v4 = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.8,
  'input_dropout_rate': 0.2,
  'initializer_range': 0.1
}

small_lstm_v5 = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.6,
  'input_dropout_rate': 0.2,
  'initializer_range': 0.1
}

small_lstm_v6 = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.8,
  'input_dropout_rate': 0.25,
  'initializer_range': 0.1
}

tiny_lstm = {
  'hidden_dim': 128,
  'embedding_dim': 128,
  'depth': 2,
  'hidden_dropout_rate': 0.1,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

tiny_lstm_v2 = {
  'hidden_dim': 128,
  'embedding_dim': 128,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
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

lstm_drop31_v2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.2,
}

biglstm_drop31_v2 = {
  'hidden_dim': 1024,
  'embedding_dim': 1024,
  'depth': 2,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.2,
}

lstm_drop31_v3 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.6,
  'input_dropout_rate': 0.2,
}

biglstm_drop31_v3 = {
  'hidden_dim': 1024,
  'embedding_dim': 1024,
  'depth': 2,
  'hidden_dropout_rate': 0.6,
  'input_dropout_rate': 0.25,
}

lstm_drop31_v4 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.4,
  'input_dropout_rate': 0.2,
}

lstm_drop31_v5 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 3,
  'hidden_dropout_rate': 0.4,
  'input_dropout_rate': 0.2,
}

lstm3_drop30 = {
  'hidden_dim': 512,
  'embedding_dim': 256,
  'depth': 3,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.0,
}

lstm_drop30_v2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

lstm_drop30_v3 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

lstm_drop30_v4 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.05
}

lstm3_drop60 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.6,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.05
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

lstm2_drop20 = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.1
}

lstm3_drop50 = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 3,
  'hidden_dropout_rate': 0.5,
  'input_dropout_rate': 0.1,
  'initializer_range': 0.05
}


lstm3_drop41 = {
  'hidden_dim': 256,
  'embedding_dim': 256,
  'depth': 3,
  'hidden_dropout_rate': 0.4,
  'input_dropout_rate': 0.1,
  'initializer_range': 0.05
}

lstm2_big_drop20_v2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.2,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.01
}

lstm2_big_drop30_v2 = {
  'hidden_dim': 512,
  'embedding_dim': 512,
  'depth': 2,
  'hidden_dropout_rate': 0.3,
  'input_dropout_rate': 0.0,
  'initializer_range': 0.01
}

ff_mnist = {'hidden_dim': 256,
                   'depth': 3,
                   'hidden_dropout_rate': 0.5,
                   'input_dropout_rate': 0.2}

ff_mnist1 = {'hidden_dim': [512, 256, 128],
                   'depth': 3,
                   'hidden_dropout_rate': 0.3,
                   'input_dropout_rate': 0.0}

ff_mnist2 = {'hidden_dim': [1024, 256, 64],
                   'depth': 3,
                   'hidden_dropout_rate': 0.3,
                   'input_dropout_rate': 0.0}

ff_mnist3 = {'hidden_dim': [512, 128, 64],
                   'depth': 3,
                   'hidden_dropout_rate': 0.3,
                   'input_dropout_rate': 0.0}

ff_mnist4 = {'hidden_dim': [512, 128, 32],
                   'depth': 3,
                   'hidden_dropout_rate': 0.1,
                   'input_dropout_rate': 0.0}

ff_mnist5 = {'hidden_dim': [512, 512, 64, 32],
                   'depth': 4,
                   'hidden_dropout_rate': 0.2,
                   'input_dropout_rate': 0.0}

ff_svhn = {'hidden_dim': 512,
                   'depth': 3,
                   'hidden_dropout_rate': 0.5,
                   'input_dropout_rate': 0.0}

ff_svhn2 = {'hidden_dim': 512,
                   'depth': 3,
                   'hidden_dropout_rate': 0.2,
                   'input_dropout_rate': 0.0}

ff_svhn3 = {'hidden_dim': 256,
                   'depth': 3,
                   'hidden_dropout_rate': 0.2,
                   'input_dropout_rate': 0.0}

ff_svhn4 = {'hidden_dim': 128,
                   'depth': 3,
                   'hidden_dropout_rate': 0.2,
                   'input_dropout_rate': 0.0}

vcnn_mnist1 = {
               'fc_dim': [128],
               'depth': 3,
               'proj_depth': 1,
               'filters': [128, 64, 32],
               'maxout_size': [128, 64, 32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.1,
              }

vcnn_mnist2 = {
              'fc_dim': [128, 128],
               'depth': 2,
               'proj_depth': 2,
               'filters': [64, 64, 64],
               'maxout_size': [64, 64, 64],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.3,
               'input_dropout_rate': 0.1}

vcnn_mnist3 = {
               'fc_dim': [],
               'depth': 3,
               'proj_depth': 0,
               'filters': [128, 64, 32],
               'maxout_size': [128, 64, 32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.1}

vcnn_mnist4 = {
               'fc_dim': [128],
               'depth': 3,
               'proj_depth': 1,
               'filters': [128, 64, 32],
               'maxout_size': [128, 64, 32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.0}

vcnn_mnist5 = {
               'fc_dim': [],
               'depth': 3,
               'proj_depth': 0,
               'filters': [128, 64, 64],
               'maxout_size': [128, 64, 16],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.0}

vcnn_mnist6 = {
               'fc_dim': [],
               'depth': 3,
               'proj_depth': 0,
               'filters': [128, 64, 64],
               'maxout_size': [128, 64, 8],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.0}

vcnn_mnist7 = {
               'fc_dim': [],
               'depth': 3,
               'proj_depth': 0,
               'filters': [128, 64, 64],
               'maxout_size': [128, 64, 16],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.3,
               'input_dropout_rate': 0.0}

vcnn_mnist8 = {
               'fc_dim': [],
               'depth': 3,
               'proj_depth': 0,
               'filters': [128, 64, 64],
               'maxout_size': [128, 64, 8],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(1,1), (2,2), (2,2)],
               'hidden_dropout_rate': 0.3,
               'input_dropout_rate': 0.0}

vcnn_lenet5 = {'hidden_dim': [128, 128],
               'depth': 2,
               'proj_depth': 2,
               'filters': [16, 16],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.8,
               'input_dropout_rate': 0.25}


vcnn_svhn1 = {'hidden_dim': [256, 256],
               'depth': 3,
               'proj_depth': 2,
               'filters': [32, 32, 32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.0}

vcnn_svhn2 = {'hidden_dim': [256, 256],
               'depth': 3,
               'proj_depth': 2,
               'filters': [32, 32,32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.1,
               'input_dropout_rate': 0.0}

vcnn_svhn3 = {'hidden_dim': [256, 256],
               'depth': 3,
               'proj_depth': 2,
               'filters': [32, 32,32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.1}

vcnn_svhn4 = {'hidden_dim': [256, 256],
               'depth': 3,
               'proj_depth': 2,
               'filters': [32, 32,32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.1,
               'input_dropout_rate': 0.1}

vcnn_svhn5 = {'hidden_dim': [512, 512],
               'depth': 3,
               'proj_depth': 2,
               'filters': [32, 32, 32],
               'kernel_size': [(3,3), (3,3), (3,3)],
               'pool_size': [(2,2), (2,2), (2,2)],
               'hidden_dropout_rate': 0.2,
               'input_dropout_rate': 0.0}

rsnt_svhn1 = {'hidden_dim': 512,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.2,
              'input_dropout_rate': 0.0,
              'num_res_net_blocks': 2}

rsnt_svhn2 = {'hidden_dim': 512,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.25,
              'input_dropout_rate': 0.1,
              'num_res_net_blocks': 2}

rsnt_svhn3 = {'hidden_dim': 512,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.5,
              'input_dropout_rate': 0.2,
              'num_res_net_blocks': 3}

rsnt_svhn4 = {'hidden_dim': 512,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.4,
              'input_dropout_rate': 0.1,
              'num_res_net_blocks': 3}

rsnt_svhn5 = {'hidden_dim': 128,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.3,
              'input_dropout_rate': 0.0,
              'num_res_net_blocks': 3}

rsnt_mnist1 = {'hidden_dim': 512,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.2,
              'input_dropout_rate': 0.0,
              'num_res_net_blocks': 2}

rsnt_mnist2 = {'hidden_dim': 512,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.2,
              'input_dropout_rate': 0.0,
              'num_res_net_blocks': 3}

rsnt_mnist3 = {'hidden_dim': 128,
              'pool_size': 3,
              'filters': [32, 32, 32, 32],
              'kernel_size': [(3,3), (3,3), (3,3), (3,3)],
              'hidden_dropout_rate': 0.2,
              'input_dropout_rate': 0.0,
              'num_res_net_blocks': 3}

caps_base = {'hidden_dim': 16,
               'routing': 3,
               'filters': 10,
               'hidden_dropout_rate': 0.5,
               'input_dropout_rate': 0.2}

mat_caps_base = {'A':32,
                 'B':32,
                 'C':32,
                 'D':32,
                 'epsilon':1e-9,
                 'l2':0.0000002,
                 'final_lambda':0.01,
                 'iter_routing':2}

MODEL_CONFIGS = {
  'base':{},
  'small_lstm':small_lstm,
  'tiny_lstm': tiny_lstm,
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
  'big_gpt_v2': big_gpt_v2,
  'very_big_gpt': very_big_gpt,
  'lstm_drop30_v2': lstm_drop30_v2,
  'lstm3_drop20': lstm3_drop20,
  'very_big_gpt_v2': very_big_gpt_v2,
  'small_gpt_v2': small_gpt_v2,
  'very_big_gpt_v3': very_big_gpt_v3,
  'lstm3_big_drop2': lstm3_big_drop2,
  'very_big_gpt_v4': very_big_gpt_v4,
  'very_big_gpt_v5': very_big_gpt_v5,
  'very_big_gpt_v6': very_big_gpt_v6,
  'lstm2_big_drop20': lstm2_big_drop20,
  'very_big_gpt_v7': very_big_gpt_v7,
  'lstm2_big_drop20_v2': lstm2_big_drop20_v2,
  'very_big_gpt_v8': very_big_gpt_v8,
  'lstm2_big_drop30_v2': lstm2_big_drop30_v2,
  'lstm2_drop20': lstm2_drop20,
  'tiny_lstm_v2': tiny_lstm_v2,
  'lstm3_drop30': lstm3_drop30,
  'small_lstm_v2': small_lstm_v2,
  'lstm_drop30_v3': lstm_drop30_v3,
  'lstm_drop30_v4': lstm_drop30_v4,
  'big_gpt_v3': big_gpt_v3,
  'small_gpt_v3': small_gpt_v3,
  'big_gpt_v4': big_gpt_v4,
  'small_gpt_v4': small_gpt_v4,
  'small_gpt_v5': small_gpt_v5,
  'very_big_gpt_v9': very_big_gpt_v9,
  'small_gpt_v6': small_gpt_v6,
  'small_lstm_v3': small_lstm_v3,
  'lstm3_drop60': lstm3_drop60,
  'small_gpt_v7': small_gpt_v7,
  'small_gpt_v8': small_gpt_v8,
  'small_gpt_v9': small_gpt_v9,
  'small_ugpt_v9': small_ugpt_v9,
  'small_lstm_v4': small_lstm_v4,
  'big_gpt_v9': big_gpt_v9,
  'very_big_gpt_v10': very_big_gpt_v10,
  'lstm3_drop50': lstm3_drop50,
  'lstm3_drop41': lstm3_drop41,
  'lstm_drop31_v2': lstm_drop31_v2,
  'big_gpt_v5': big_gpt_v5,
  'lstm_drop31_v3': lstm_drop31_v3,
  'big_gpt_v6': big_gpt_v6,
  'lstm_drop31_v4': lstm_drop31_v4,
  'lstm_drop31_v5': lstm_drop31_v5,
  'biglstm_drop31_v2': biglstm_drop31_v2,
  'short_gpt_v9': short_gpt_v9,
  'ff_mnist': ff_mnist,
  'vcnn_mnist1': vcnn_mnist1,
  'vcnn_mnist2': vcnn_mnist2,
  'vcnn_mnist3': vcnn_mnist3,
  'vcnn_mnist5': vcnn_mnist5,
  'vcnn_mnist6': vcnn_mnist6,
  'vcnn_mnist7': vcnn_mnist7,
  'vcnn_mnist8': vcnn_mnist8,
  'vcnn_mnist4': vcnn_mnist4,
  'caps_base': caps_base,
  'biglstm_drop31_v3': biglstm_drop31_v3,
  'mat_caps_base': mat_caps_base,
  'small_lstm_v6': small_lstm_v6,
  'vcnn_svhn1': vcnn_svhn1,
  'vcnn_svhn2': vcnn_svhn2,
  'vcnn_svhn3': vcnn_svhn3,
  'vcnn_svhn4': vcnn_svhn4,
  'vcnn_svhn5': vcnn_svhn5,
  'rsnt_svhn1': rsnt_svhn1,
  'rsnt_svhn2': rsnt_svhn2,
  'rsnt_svhn3': rsnt_svhn3,
  'rsnt_svhn4': rsnt_svhn4,
  'rsnt_svhn5': rsnt_svhn5,
  'ff_svhn': ff_svhn,
  'ff_svhn2': ff_svhn2,
  'ff_svhn3': ff_svhn3,
  'ff_svhn4': ff_svhn4,
  'rsnt_mnist1': rsnt_mnist1,
  'rsnt_mnist2': rsnt_mnist2,
  'rsnt_mnist3': rsnt_mnist3,
  'ff_mnist1': ff_mnist1,
  'ff_mnist2': ff_mnist2,
  'ff_mnist3': ff_mnist3,
  'ff_mnist4': ff_mnist4,
  'ff_mnist5': ff_mnist5
  }