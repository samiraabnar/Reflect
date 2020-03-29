import tensorflow as tf
from tf2_models.common_layers import get_initializer, shape_list
from tf2_models.embedding import SharedEmbeddings
from tf2_models.transformer_layers import Block
from tf2_models.transformers import *

class LmGPT2(tf.keras.Model):
  def __init__(self, hparams, scope='lm_gpt2', *inputs, **kwargs):
    del kwargs['cl_token']
    super(LmGPT2, self).__init__(hparams, *inputs, **kwargs)
    self.scope = scope
    self.rep_index = 1
    self.rep_layer = None

    self.model_name = '_'.join([self.scope,
                         'h-'+str(hparams.embedding_dim),
                         'd-'+str(hparams.depth),
                         'rdrop-'+str(hparams.resid_pdrop),
                         'adrop-' + str(hparams.attn_pdrop),
                         'indrop-'+str(hparams.embd_pdrop)])

    self.create_vars(hparams)

  @tf.function
  def create_vars(self, hparams):
    self.transformer = GPT2(hparams, name='transformer')

  def call(self, inputs, **kwargs):
    transformer_outputs = self.transformer(inputs, **kwargs)
    hidden_states = transformer_outputs[0]

    lm_logits = self.transformer.wte(hidden_states, mode="linear")

    #outputs = (lm_logits,) + transformer_outputs[1:]

    return lm_logits  # lm_logits, presents, (all hidden_states), (attentions)

  def detailed_call(self, inputs, **kwargs):
    transformer_outputs = self.transformer(inputs, **kwargs)
    hidden_states = transformer_outputs[0]

    lm_logits = self.transformer.wte(hidden_states, mode="linear")

    outputs = (lm_logits,) + transformer_outputs
    return outputs  # lm_logits, presents, (all hidden_states), (attentions)

class LmGPT2SharedWeights(LmGPT2):
  def __init__(self, hparams, scope='lm_gpt2_shared_weights', *inputs, **kwargs):
    super(LmGPT2SharedWeights, self).__init__(hparams, *inputs, **kwargs)

  @tf.function
  def create_vars(self, hparams):
    self.transformer = GPT2SharedWeights(hparams, name='shared_transformer')

  def call(self, inputs, **kwargs):
    transformer_outputs = self.transformer(inputs, **kwargs)
    hidden_states = transformer_outputs[0]

    lm_logits = self.transformer.wte(hidden_states, mode="linear")

    #outputs = (lm_logits,) + transformer_outputs[1:]

    return lm_logits  # lm_logits, presents, (all hidden_states), (attentions)

class ClassifierGPT2(tf.keras.Model):
  def __init__(self, hparams, scope='cl_gpt2',*inputs, **kwargs):
    self.cl_token = kwargs['cl_token']
    del kwargs['cl_token']
    super(ClassifierGPT2, self).__init__(hparams, *inputs, **kwargs)
    self.rep_index = 2
    self.rep_layer = None

    self.scope = scope
    self.hparams = hparams
    self.model_name = '_'.join([self.scope,
                         'h-'+str(hparams.embedding_dim),
                         'd-'+str(hparams.depth),
                         'rdrop-'+str(hparams.resid_pdrop),
                         'adrop-' + str(hparams.attn_pdrop),
                         'indrop-'+str(hparams.embd_pdrop)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.0001)
    self.create_vars(**kwargs)

  #@tf.function
  def create_vars(self,**kwargs):
    self.transformer = GPT2(self.hparams, name='transformer',
                            **kwargs)
    self.e2c = tf.keras.layers.Dense(units=self.hparams.num_labels,
                                     kernel_initializer=get_initializer(self.hparams.initializer_range),
                                     name='e2c')

  def call(self, inputs, **kwargs):
    @tf.function(experimental_relax_shapes=True)
    def _call(batch_size, inputs, transformer_outputs):
      mask = tf.cast(inputs != 0, dtype=tf.int32)
      inputs_lengths = tf.reduce_sum(mask, axis=-1) - 1
      batch_indices = tf.range(batch_size)
      indices = tf.concat([batch_indices[..., None], inputs_lengths[..., None]], -1)
      hidden_states = tf.gather_nd(transformer_outputs[0], indices)
      cl_logits = self.e2c(hidden_states)
      return cl_logits

    # Add CL token:
    batch_size = tf.shape(inputs)[0]
    #cl_token = tf.reshape(tf.convert_to_tensor(self.cl_token[0], dtype=tf.int64)[None], (-1,1))
    #cl_tokens = tf.tile(cl_token, (batch_size, 1))
    #inputs = tf.concat([cl_tokens, inputs], axis=-1)

    transformer_outputs = self.transformer(inputs, **kwargs)
    cl_logits = _call(batch_size, inputs, transformer_outputs)


    return cl_logits

  def detailed_call(self, inputs, **kwargs):
    @tf.function(experimental_relax_shapes=True)
    def _call(batch_size, inputs, transformer_outputs):
      mask = tf.cast(inputs != 0, dtype=tf.int32)
      inputs_lengths = tf.reduce_sum(mask, axis=-1) - 1
      batch_indices = tf.range(batch_size)
      indices = tf.concat([batch_indices[..., None], inputs_lengths[..., None]], -1)
      hidden_states = tf.gather_nd(transformer_outputs[0], indices)
      cl_logits = self.e2c(hidden_states)
      return cl_logits, hidden_states

    # Add CL token:
    batch_size = tf.shape(inputs)[0]
    #cl_token = tf.reshape(tf.convert_to_tensor(self.cl_token[0], dtype=tf.int64)[None], (-1,1))
    #cl_tokens = tf.tile(cl_token, (batch_size, 1))
    #inputs = tf.concat([cl_tokens, inputs], axis=-1)

    transformer_outputs = self.transformer(inputs, **kwargs)
    cl_logits, hidden_states = _call(batch_size, inputs, transformer_outputs)

    outputs = (cl_logits, hidden_states) + transformer_outputs

    return outputs

class ClassifierGPT2SharedWeights(ClassifierGPT2):
  def __init__(self, hparams, scope='cl_gpt2_shared_weights', *inputs, **kwargs):
    super(ClassifierGPT2SharedWeights, self).__init__(hparams, scope=scope, *inputs, **kwargs)

  @tf.function
  def create_vars(self):
    self.transformer = GPT2SharedWeights(self.hparams, name='shared_transformer')
    self.e2c = tf.keras.layers.Dense(units=self.hparams.num_labels,
                                     kernel_initializer=get_initializer(self.hparams.initializer_range),
                                     name='e2c')

class ClassifierBERT(tf.keras.Model):
  def __init__(self, hparams, scope='cl_bert',*inputs, **kwargs):
    self.cl_token = kwargs['cl_token']
    del kwargs['cl_token']
    super(ClassifierBERT, self).__init__(hparams, *inputs, **kwargs)

    self.scope = scope
    self.hparams = hparams
    self.rep_index = 2
    self.rep_layer = None
    self.model_name = '_'.join([self.scope,
                         'h-'+str(hparams.embedding_dim),
                         'd-'+str(hparams.depth),
                         'rdrop-'+str(hparams.resid_pdrop),
                         'adrop-' + str(hparams.attn_pdrop),
                         'indrop-'+str(hparams.embd_pdrop)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.0001)
    self.create_vars(**kwargs)

  #@tf.function
  def create_vars(self,**kwargs):
    self.transformer = Bert(self.hparams, name='transformer',
                            **kwargs)
    self.e2c = tf.keras.layers.Dense(units=self.hparams.num_labels,
                                     kernel_initializer=get_initializer(self.hparams.initializer_range),
                                     name='e2c')

  def call(self, inputs, **kwargs):
    @tf.function(experimental_relax_shapes=True)
    def _call(batch_size, inputs, transformer_outputs):
      #mask = tf.cast(inputs != 0, dtype=tf.int32)
      #inputs_lengths = tf.reduce_sum(mask, axis=-1) - 1
      #batch_indices = tf.range(batch_size)
      #indices = tf.concat([batch_indices[..., None], inputs_lengths[..., None]], -1)
      hidden_states = transformer_outputs[0][:,0]#tf.gather_nd(transformer_outputs[0], indices)
      cl_logits = self.e2c(hidden_states, **kwargs)
      return cl_logits

    # Add CL token:
    batch_size = tf.shape(inputs)[0]
    cl_token = tf.reshape(tf.convert_to_tensor(self.cl_token[0], dtype=tf.int64)[None], (-1,1))
    cl_tokens = tf.tile(cl_token, (batch_size, 1))
    inputs = tf.concat([cl_tokens, inputs], axis=-1)

    transformer_outputs = self.transformer(inputs, **kwargs)
    cl_logits = _call(batch_size, inputs, transformer_outputs)


    return cl_logits

  def detailed_call(self, inputs, **kwargs):
    @tf.function(experimental_relax_shapes=True)
    def _call(batch_size, inputs, transformer_outputs):
      hidden_states = transformer_outputs[0][:, 0]
      cl_logits = self.e2c(hidden_states)
      return cl_logits, hidden_states

    # Add CL token:
    batch_size = tf.shape(inputs)[0]
    cl_token = tf.reshape(tf.convert_to_tensor(self.cl_token[0], dtype=tf.int64)[None], (-1,1))
    cl_tokens = tf.tile(cl_token, (batch_size, 1))
    inputs = tf.concat([cl_tokens, inputs], axis=-1)

    transformer_outputs = self.transformer(inputs, **kwargs)
    cl_logits, hidden_states = _call(batch_size, inputs, transformer_outputs)

    outputs = (cl_logits, hidden_states, transformer_outputs[0][:,1:,:]) + transformer_outputs

    return outputs

  def get_input_embeddings(self, inputs, **kwargs):
    # Add CL token:
    batch_size = tf.shape(inputs)[0]
    cl_token = tf.reshape(tf.convert_to_tensor(self.cl_token[0], dtype=tf.int64)[None], (-1, 1))
    cl_tokens = tf.tile(cl_token, (batch_size, 1))
    inputs = tf.concat([cl_tokens, inputs], axis=-1)

    outputs = self.transformer.get_input_embeddings(inputs, **kwargs)

    return outputs

  def call_with_embeddings(self, input_embeddings, input_shape, padding_mask, past , **kwargs):


    outputs = self.transformer.call_with_embeddings(input_embeddings=input_embeddings,
                                                    input_shape=input_shape, padding_mask=padding_mask,
                                                    past=past, **kwargs)
    return outputs

class ClassifierBERTSharedWeights(ClassifierBERT):
  def __init__(self, hparams, scope='cl_bert_shared', *inputs, **kwargs):
    super(ClassifierBERTSharedWeights, self).__init__(hparams, scope=scope, *inputs, **kwargs)


  # @tf.function
  def create_vars(self, **kwargs):
    self.transformer = BertSharedWeights(self.hparams, name='transformer',
                            **kwargs)
    self.e2c = tf.keras.layers.Dense(units=self.hparams.num_labels,
                                     kernel_initializer=get_initializer(self.hparams.initializer_range),
                                     name='e2c')

