import absl
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export

from tensorflow_addons.utils import keras_utils


@keras_export("keras.optimizers.schedules.ExponentialDecay")
class ExponentialDecayWithWarmpUp(LearningRateSchedule):
  """A LearningRateSchedule that uses an exponential decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      warmup_steps,
      warmup_learning_rate=0.0,
      hold_base_rate_steps=0,
      staircase=False,
      name=None):
    """Applies exponential decay to the learning rate.

    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies an exponential decay function
    to an optimizer step, given a provided initial learning rate.

    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate * decay_rate ^ (step / decay_steps)
    ```

    If the argument `staircase` is `True`, then `step / decay_steps` is
    an integer division and the decayed learning rate follows a
    staircase function.

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.
    Example: When fitting a Keras model, decay every 100000 steps with a base
    of 0.96:

    ```python
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      decay_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The decay rate.
      staircase: Boolean.  If `True` decay the learning rate at discrete
        intervals
      name: String.  Optional name of the operation.  Defaults to
        'ExponentialDecay'.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    super(ExponentialDecayWithWarmpUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.warmup_steps = warmup_steps
    self.warmup_learning_rate = warmup_learning_rate
    self.hold_base_rate_steps = hold_base_rate_steps
    self.staircase = staircase
    self.name = name

  @tf.function(experimental_relax_shapes=True)
  def __call__(self, step):
    with ops.name_scope_v2(self.name or "ExponentialDecay") as name:
      initial_learning_rate = tf.constant(
          self.initial_learning_rate, name="initial_learning_rate", dtype=tf.float32)
      warmup_learning_rate =  tf.constant(
          self.warmup_learning_rate, name="warmup_learning_rate", dtype=tf.float32)
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)
      warmup_steps = math_ops.cast(self.warmup_steps, dtype)
      hold_base_rate_steps = math_ops.cast(self.hold_base_rate_steps, dtype)
      global_step_recomp = math_ops.cast(step, dtype)


      p = (global_step_recomp - (warmup_steps+hold_base_rate_steps)) / (decay_steps)
      if self.staircase:
          p = math_ops.floor(p)
      learning_rate= math_ops.multiply(
          initial_learning_rate, math_ops.pow(decay_rate, p), name=name)

      learning_rate = tf.where(
          global_step_recomp > (warmup_steps + hold_base_rate_steps),
          learning_rate, initial_learning_rate)

      if self.warmup_steps > 0:
          if self.initial_learning_rate < self.warmup_learning_rate:
              raise ValueError('learning_rate_base must be larger or equal to '
                               'warmup_learning_rate.')
          slope = (initial_learning_rate - warmup_learning_rate) / warmup_steps
          warmup_rate = slope * tf.cast(global_step_recomp,
                                        tf.float32) + warmup_learning_rate

          learning_rate = tf.where(global_step_recomp < warmup_steps, warmup_rate,
                                   learning_rate)

    return learning_rate


  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "warmup_steps": self.warmup_steps,
        "warmup_learning_rate": self.warmup_learning_rate,
        "hold_base_rate_steps": self.hold_base_rate_steps,
        "name": self.name
    }


class RectifiedAdam(tf.keras.optimizers.Optimizer):
    """Variant of the Adam optimizer whose adaptive learning rate is rectified
    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).
    Example of usage:
    ```python
    opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
    ```
    Note: `amsgrad` is not described in the original paper. Use it with
          caution.
    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:
    ```python
    opt = tfa.optimizers.RectifiedAdam(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
    )
    ```
    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.
    Lookahead, proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
    which is announced by Less Wright and the new combined optimizer can also
    be called "Ranger". The mechanism can be enabled by using the lookahead
    wrapper. For example:
    ```python
    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ```
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=0.,
                 amsgrad=False,
                 sma_threshold=5.0,
                 total_steps=0,
                 warmup_proportion=0.1,
                 min_lr=0.,
                 name='RectifiedAdam',
                 **kwargs):
        r"""Construct a new RAdam optimizer.
        Args:
            learning_rate: A Tensor or a floating point value.
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            sma_threshold. A float value.
                The threshold for simple mean average.
            total_steps: An integer. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value.
                The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying
                gradients. Defaults to "RectifiedAdam".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super(RectifiedAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('sma_threshold', sma_threshold)
        self._set_hyper('total_steps', float(total_steps))
        self._set_hyper('warmup_proportion', warmup_proportion)
        self._set_hyper('min_lr', min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._initial_weight_decay = weight_decay
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(RectifiedAdam, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper('total_steps', var_dtype)
            warmup_steps = total_steps *\
                self._get_hyper('warmup_proportion', var_dtype)
            min_lr = self._get_hyper('min_lr', var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps,
                                               decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (
            1.0 - beta_2_power)

        m_t = m.assign(
            beta_1_t * m + (1.0 - beta_1_t) * grad,
            use_locking=self._use_locking)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v_t = v.assign(
            beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad),
            use_locking=self._use_locking)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = vhat.assign(
                tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        r_t = tf.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) /
                      (sma_inf - 2.0) * sma_inf / sma_t)

        sma_threshold = self._get_hyper('sma_threshold', var_dtype)
        var_t = tf.where(sma_t >= sma_threshold,
                         r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t)

        if self._initial_weight_decay > 0.0:
            var_t += self._get_hyper('weight_decay', var_dtype) * var

        var_update = var.assign_sub(
            lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper('total_steps', var_dtype)
            warmup_steps = total_steps *\
                self._get_hyper('warmup_proportion', var_dtype)
            min_lr = self._get_hyper('min_lr', var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps,
                                               decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (
            1.0 - beta_2_power)

        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = v.assign(v * beta_2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = vhat.assign(
                tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        r_t = tf.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) /
                      (sma_inf - 2.0) * sma_inf / sma_t)

        sma_threshold = self._get_hyper('sma_threshold', var_dtype)
        var_t = tf.where(sma_t >= sma_threshold,
                         r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t)

        if self._initial_weight_decay > 0.0:
            var_t += self._get_hyper('weight_decay', var_dtype) * var

        with tf.control_dependencies([var_t]):
            var_update = self._resource_scatter_add(
                var, indices, tf.gather(-lr_t * var_t, indices))

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super(RectifiedAdam, self).get_config()
        config.update({
            'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
            'beta_1':
            self._serialize_hyperparameter('beta_1'),
            'beta_2':
            self._serialize_hyperparameter('beta_2'),
            'decay':
            self._serialize_hyperparameter('decay'),
            'weight_decay':
            self._serialize_hyperparameter('weight_decay'),
            'sma_threshold':
            self._serialize_hyperparameter('sma_threshold'),
            'epsilon':
            self.epsilon,
            'amsgrad':
            self.amsgrad,
            'total_steps':
            self._serialize_hyperparameter('total_steps'),
            'warmup_proportion':
            self._serialize_hyperparameter('warmup_proportion'),
            'min_lr':
            self._serialize_hyperparameter('min_lr'),
        })
        return config