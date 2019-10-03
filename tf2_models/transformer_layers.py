import tensorflow as tf

from tf2_models.common_layers import get_initializer, shape_list, gelu


class Attention(tf.keras.layers.Layer):
  def __init__(self, hidden_dim, n_ctx, config, regularizer, scale=False, **kwargs):
    super(Attention, self).__init__(**kwargs)
    self.output_attentions = config.output_attentions

    n_state = hidden_dim
    assert n_state % config.n_head == 0

    self.n_ctx = n_ctx
    self.n_head = config.n_head
    self.split_size = n_state
    self.scale = scale
    self.regularizer = regularizer
    self.c_attn = Conv1D(nf=n_state * 3, nx=hidden_dim,
                         initializer_range=config.initializer_range,
                         regularizer=self.regularizer, name='c_attn')
    self.c_proj = Conv1D(nf=n_state, nx=hidden_dim,
                         initializer_range=config.initializer_range,
                         regularizer=self.regularizer,
                         name='c_proj')
    self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
    self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)

  @staticmethod
  def causal_attention_mask(nd, ns, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

  def _attn(self, inputs, training=False):
    q, k, v, attention_mask = inputs
    # q, k, v have shape [batch, heads, sequence, features]
    w = tf.matmul(q, k, transpose_b=True)
    if self.scale:
      dk = tf.cast(tf.shape(k)[-1], tf.float32)  # scale attention_scores
      w = w / tf.math.sqrt(dk)

    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    _, _, nd, ns = shape_list(w)
    b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
    b = tf.reshape(b, [1, 1, nd, ns])
    w = w * b - 1e4 * (1 - b)

    if attention_mask is not None:
      # Apply the attention mask
      w = w + attention_mask

    w = tf.nn.softmax(w, axis=-1)
    w = self.attn_dropout(w, training=training)

    outputs = [tf.matmul(w, v)]
    if self.output_attentions:
      outputs.append(w)
    return outputs

  def merge_heads(self, x):
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
    return tf.reshape(x, new_x_shape)

  def split_heads(self, x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
    x = tf.reshape(x, new_x_shape)
    return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

  def call(self, inputs, training=False):
    x, layer_past, attention_mask = inputs

    x = self.c_attn(x)
    query, key, value = tf.split(x, 3, axis=2)
    query = self.split_heads(query)
    key = self.split_heads(key)
    value = self.split_heads(value)
    if layer_past is not None:
      past_key, past_value = tf.unstack(layer_past, axis=1)
      key = tf.concat([past_key, key], axis=-2)
      value = tf.concat([past_value, value], axis=-2)
    present = tf.stack([key, value], axis=1)

    attn_outputs = self._attn([query, key, value, attention_mask], training=training)
    a = attn_outputs[0]

    a = self.merge_heads(a)
    a = self.c_proj(a)
    a = self.resid_dropout(a, training=training)

    outputs = [a, present] + attn_outputs[1:]
    return outputs  # a, present, (attentions)


class Conv1D(tf.keras.layers.Layer):
  def __init__(self, nf, nx, regularizer, initializer_range=0.02, **kwargs):
    """ TFConv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
        Basically works like a Linear layer but the weights are transposed
    """
    super(Conv1D, self).__init__(**kwargs)
    self.nf = nf
    self.nx = nx
    self.initializer_range = initializer_range
    self.regularizer = regularizer

  def build(self, input_shape):
    self.weight = self.add_weight(
      "weight",
      shape=[self.nx, self.nf],
      initializer=get_initializer(self.initializer_range),
      regularizer=self.regularizer)
    self.bias = self.add_weight(
      "bias",
      shape=[1, self.nf],
      initializer=tf.zeros_initializer(),
      regularizer=self.regularizer)

  def call(self, x, **kwargs):
    bz, sl = shape_list(x)[:2]

    x = tf.reshape(x, [-1, self.nx])
    x = tf.matmul(x, self.weight) + self.bias

    x = tf.reshape(x, [bz, sl, self.nf])

    return x


class Block(tf.keras.layers.Layer):
  def __init__(self, n_ctx, config, regularizer, scale=False, **kwargs):
    super(Block, self).__init__(**kwargs)
    self.regularizer = regularizer
    nx = config.embedding_dim
    self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_1')
    self.attn = Attention(hidden_dim=nx, n_ctx=n_ctx, config=config, scale=scale, regularizer=self.regularizer, name='attn')
    self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_2')
    self.mlp = TransformerMLP(4 * nx, config, regularizer=self.regularizer, name='mlp')

  def call(self, inputs, training=False):
    x, layer_past, attention_mask = inputs

    a = self.ln_1(x)
    output_attn = self.attn([a, layer_past, attention_mask], training=training)
    a = output_attn[0]  # output_attn: a, present, (attentions)
    x = x + a

    m = self.ln_2(x)
    m = self.mlp(m, training=training)
    x = x + m

    outputs = [x] + output_attn[1:]
    return outputs  # x, present, (attentions)


class TransformerMLP(tf.keras.layers.Layer):
  def __init__(self, n_state, config, regularizer, **kwargs):
    super(TransformerMLP, self).__init__(**kwargs)
    self.regularizer = regularizer
    nx = config.embedding_dim
    self.c_fc = Conv1D(n_state, nx, initializer_range=config.initializer_range,
                       regularizer=self.regularizer, name='c_fc')
    self.c_proj = Conv1D(nx, n_state, initializer_range=config.initializer_range,
                         regularizer=self.regularizer, name='c_proj')
    self.act = gelu
    self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)

  def call(self, x, training=False):
    h = self.act(self.c_fc(x))
    h2 = self.c_proj(h)
    h2 = self.dropout(h2, training=training)
    return h2