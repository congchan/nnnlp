" Tensorflow version 1.x modeling codes. "
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import math
import collections
import re
import six
from six.moves import range
import tensorflow as tf


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def get_assignment_map_from_albert_checkpoint(tvars, init_checkpoint, num_of_group=0):
  """Compute the union of the current variables and albert checkpoint variables.
     albert contains groups of layers which share variables.
  """
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var
  init_vars = tf.train.list_variables(init_checkpoint)
  init_vars_name = [name for (name, _) in init_vars]

  if num_of_group > 0:
    assignment_map = []
    for gid in range(num_of_group):
      assignment_map.append(collections.OrderedDict())
  else:
    assignment_map = collections.OrderedDict()

  for name in name_to_variable:
    if name in init_vars_name:
      tvar_name = name
    elif (re.sub(r"/group_\d+/", "/group_0/",
                 six.ensure_str(name)) in init_vars_name and
          num_of_group > 1):
      tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
    elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
          in init_vars_name and num_of_group > 1):
      tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
    elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name))
          in init_vars_name and num_of_group > 1):
      tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
                         six.ensure_str(name))
    else:
      tf.logging.info("name %s does not get matched", name)
      continue
    tf.logging.info("name %s match to %s", name, tvar_name)
    if num_of_group > 0:
      group_matched = False
      for gid in range(1, num_of_group):
        if (("/group_" + str(gid) + "/" in name) or
            ("/ffn_" + str(gid) + "/" in name) or
            ("/attention_" + str(gid) + "/" in name)):
          group_matched = True
          tf.logging.info("%s belongs to %dth", name, gid)
          assignment_map[gid][tvar_name] = name
      if not group_matched:
        assignment_map[0][tvar_name] = name
    else:
      assignment_map[tvar_name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[six.ensure_str(name) + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.
  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.
  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def crf_layer(inputs, tag_indices, num_labels, true_sequence_lengths, 
    transitions_name="transitions", inference_only=False):
  """ Performs tensorflow crf decoding.
  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary 
        potentials to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices 
        for which we compute the log-likelihood.
    num_labels: A int number indicates the number of possible tags.
    true_sequence_lengths: A [batch_size] vector of true sequence lengths.
  Returns:
    per_example_loss: A [batch_size] Tensor containing the negative
        log-likelihood of each example, given the sequence of tag indices.
    predictions: A [batch_size, max_seq_len] CRF decode_tags represent 
        the most probable tags sequence.
    best_score: A [batch_size] vector, containing the score of decode_tags.
  """
  with tf.variable_scope('crf'):
    transition_params = tf.get_variable(
        transitions_name,
        shape=[num_labels, num_labels],
        initializer=tf.zeros_initializer())
  per_example_loss = None
  if not inference_only:
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        inputs=inputs,
        tag_indices=tag_indices,
        transition_params=transition_params,
        sequence_lengths=true_sequence_lengths)
    per_example_loss = -log_likelihood
  # NOTE CRF decode, decode_tags [batch_size, max_seq_len] most probable path
  decode_tags, best_score = tf.contrib.crf.crf_decode(potentials=inputs,
      transition_params=transition_params,
      # NOTE sequence_length: [batch_size] vector of true sequence lengths.
      sequence_length=true_sequence_lengths)
  # A [batch_size] Tensor containing the -log_likelihood of each example
  predictions = decode_tags
  return per_example_loss, predictions, best_score


def softmax(logits, labels, num_classes, mask=None):
  """ Perform softmax operation
  Args:
    logits: Logits outputs of the network, last dimension should be num_classes, 
        ie.. [batch_size, seq_length ?, num_classes];
    labels: A [batch_size, seq_length ?] tensor represent true label ids.
    mask: mask should be the same shape as logits.
  Return:
    per_example_loss, a [batch_size] tensor containing the cross_entropy loss.
  """
  # # shape: batch x features_tokens x depth if axis==-1, same shape as logits
  one_hot_labels = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
  # # shape: batch x features_tokens
  # per_token_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
  #     labels=one_hot_labels)
      
  # An alternative way, less efficient, but allow self-define mask
  log_probs = tf.nn.log_softmax(logits, axis=-1) # NOTE same shape as logits
  # NOTE shape (batch_size, max_seq_len, depth)
  per_token_loss = one_hot_labels * log_probs

  if mask is not None and len(per_token_loss.shape.as_list()) > 2:
    mask = tf.cast(mask, tf.float32)
    per_token_loss = tf.einsum("BFH,BF->BFH", per_token_loss, mask)

  per_example_loss = -tf.reduce_sum(per_token_loss, axis=-1)

  probabilities = tf.nn.softmax(logits, axis=-1) # NOTE same shape as logits
  predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int64)

  return per_example_loss, predictions


def am_softmax(logits, labels, num_classes, scale=30, margin=0.35):
  """ Addictive angular softmax, which defers from regular softmax for:
  1. Perform L2 normalization on output layer's input and weight, 
      the logits, i.e. dot product between input and weight, become COS.
  2. COS value minus a constant positive margin for positive label, 
      then scale by factor scale.
  """
  with tf.name_scope('am_softmax'):
    # shape: batch x features_tokens x depth if axis == -1, same shape as logits
    one_hot_labels = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    logits = one_hot_labels * (logits - margin) + (1 - one_hot_labels) * logits
    logits *= scale
    # shape: batch x features_tokens
    per_token_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
        labels=one_hot_labels)
    probabilities = tf.nn.softmax(logits, axis=-1) # NOTE same shape as logits
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int64)      
    return per_token_loss, predictions  


def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
		similarity = y * tf.square(distance) # keep the similar label (1) close to each other
		dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0)) # give penalty to dissimilar label if the distance is bigger than margin
		return tf.reduce_mean(dissimilarity + similarity) / 2


def lstm_fused(inputs, sequence_length, lstm_size, bilstm_dropout_rate, 
    is_training, num_layers=1):
  """ FusedRNNCell
  Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len)` or None.
  Returns:
      Cell state (cs): A `3-D` tensor of shape `[time_len, batch_size,
                         output_size]`
  """
  def _lstm_fused(inputs, sequence_length, lstm_size, is_training, dropout_rate=0.5, 
                    scope='lstm-fused'):
    with tf.variable_scope(scope):
      inputs = tf.transpose(inputs, perm=[1, 0, 2])  # Need time-major
      lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)
      outputs, _ = lstm_cell(inputs, dtype=tf.float32, 
                            sequence_length=sequence_length)
      outputs = tf.transpose(outputs, perm=[1, 0, 2])
      outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
      return outputs

  rnn_output = tf.identity(inputs)
  for i in range(num_layers):
    scope = 'lstm-fused-%s' % i
    rnn_output = _lstm_fused(rnn_output, sequence_length, lstm_size=lstm_size,
                                is_training=is_training,
                                dropout_rate=bilstm_dropout_rate,
                                scope=scope)  # (batch_size, seq_length, 2*rnn_size)
  return rnn_output


def bilstm_fused(inputs, sequence_lengths, lstm_size, bilstm_dropout_rate, 
    is_training, num_layers=1):
  """ FusedRNNCell uses a single TF op for the entire LSTM. """ 
  def _bi_lstm_fused(inputs, sequence_lengths, rnn_size, is_training, 
                    dropout_rate=0.5, scope='bi-lstm-fused'):
    with tf.variable_scope(scope):
      inputs = tf.transpose(inputs, perm=[1, 0, 2])  # Need time-major
      lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
      lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
      lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
      output_fw, _ = lstm_cell_fw(inputs, dtype=tf.float32, 
          sequence_length=sequence_lengths)
      output_bw, _ = lstm_cell_bw(inputs, dtype=tf.float32, 
          sequence_length=sequence_lengths)
      outputs = tf.concat([output_fw, output_bw], axis=-1)
      outputs = tf.transpose(outputs, perm=[1, 0, 2])
      outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
      return outputs

  rnn_output = tf.identity(inputs)
  for i in range(num_layers):
    scope = 'bi-lstm-fused-%s' % i
    rnn_output = _bi_lstm_fused(rnn_output, sequence_lengths, rnn_size=lstm_size,
                                is_training=is_training,
                                dropout_rate=bilstm_dropout_rate,
                                scope=scope)  # (batch_size, seq_length, 2*rnn_size)
  return rnn_output


def idcnn_layer(config, model_inputs, name=None):
  """
  config params:
    is_train
    filter_height: for text, 1
    filter_width: 
    embedding_dim: # input channels
    num_filter: # of output channels
    repeat_times
    layers

   
  :param idcnn_inputs: [batch_size, seq_len, emb_size] 
  :return: [batch_size, seq_len, cnn_output_width]
  """
  # shape of input = [batch, in_height=1, in_width=seq_len, in_channels=emb_size]
  model_inputs = tf.expand_dims(model_inputs, 1)

  with tf.variable_scope("idcnn" if not name else name):
    # filter [filter_height, filter_width, in_channels, out_channels]
    filter_shape = [1, config.filter_width, config.embedding_dim,
                config.num_filter]
    # print(shape)
    filter_weights = tf.get_variable(
        "idcnn_filter",
        shape=filter_shape,
        initializer=config.initializer)
    layerInput = tf.nn.conv2d(model_inputs,
                              filter_weights,
                              strides=[1, 1, 1, 1],
                              padding="SAME",
                              name="init_layer")
    finalOutFromLayers = []
    totalWidthForLastDim = 0
    for j in range(config.repeat_times):
      for i in range(len(config.layers)):
        dilation = config.layers[i]['dilation']
        isLast = True if i == (len(config.layers) - 1) else False
        with tf.variable_scope("atrous-conv-layer-%d" % i,
                                reuse=tf.AUTO_REUSE):
          w = tf.get_variable(
              "filter_w",
              shape=[1, config.filter_width, config.num_filter,
                      config.num_filter],
              initializer=tf.contrib.layers.xavier_initializer())
          b = tf.get_variable("filter_b", shape=[config.num_filter])
          conv = tf.nn.atrous_conv2d(layerInput,
                                      w,
                                      rate=dilation,
                                      padding="SAME")
          conv = tf.nn.bias_add(conv, b)
          conv = tf.nn.relu(conv)
          if isLast:
            finalOutFromLayers.append(conv)
            totalWidthForLastDim += config.num_filter
          layerInput = conv
            
    finalOut = tf.concat(axis=3, values=finalOutFromLayers)
    keepProb = 1.0 if config.is_train else 0.5
    finalOut = tf.nn.dropout(finalOut, keepProb)

    finalOut = tf.squeeze(finalOut, [1])
    finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
    config.cnn_output_width = totalWidthForLastDim
    return finalOut

      
def cudnn_rnn(inputs, sequence_lengths, time_major=False,
    num_layers=1, dropout=0.0, rnn_size=128, is_training=True,
    cell_type='lstm', direction='unidirectional'):
  """ cudnn_lstm/gru/rnn for id tensor.
  Args:
    inputs: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
    sequence_lengths: an int32 array representing the variable sequence
        lengths in a batch. The size of the array has to equal the batch_size.
        If not provided, the same sequence length will be assumed.
    time_major: The shape format of the `inputs` and `outputs` Tensors. If
        true, these Tensors must be shaped ['max_time', 'batch_size', 'depth'].
        If false, these Tensors must be shaped ['batch_size', 'max_time',
        'depth']. By default this function accepts input and emits output in
        time-major form. This param is only effective when 'sequence_lengths' is
        used.
    training: whether this operation will be used in training or inference.
    direction: the direction model that the model operates. 
        Can be either 'unidirectional' or 'bidirectional'
  Returns:
      output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`
        if `time_major` is True (default) or `[batch_size, time_len,
        num_dirs * num_units]` if `time_major` is False.
        It is a `concat([fwd_output, bak_output], axis=2)`.
      output_states: a tuple of tensor(s) of the same shape and structure as
        `initial_state`.
  """
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if inputs.shape.ndims == 2:
      inputs = tf.expand_dims(inputs, axis=[-1])
  model_dic = {
      'lstm': tf.contrib.cudnn_rnn.CudnnLSTM,
      'gru': tf.contrib.cudnn_rnn.CudnnGRU,
      'rnn_relu': tf.contrib.cudnn_rnn.CudnnRNNRelu,
      'rnn_tanh': tf.contrib.cudnn_rnn.CudnnRNNTanh,
  }
  model = model_dic[cell_type]
  fn = model(
      num_layers=num_layers,
      num_units=rnn_size,
      # input_mode=CUDNN_INPUT_LINEAR_MODE,
      direction=direction,
      dropout=dropout,
      # seed=None,
      # dtype=tf.dtypes.float32,
      # kernel_initializer=None,
      # bias_initializer=None,
      # name=None
      )
  outputs, output_states = fn(
      inputs=inputs,
      # initial_state=None,
      sequence_lengths=sequence_lengths,
      time_major=time_major,
      training=is_training,)

  return outputs, output_states


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def einsum_via_matmul(input_tensor, w, num_inner_dims):
  """Implements einsum via matmul and reshape ops.
  to perform tf.einsum("BFH,HO->BFO", input_tensor, w),
    call einsum_via_matmul(input_tensor, w, 1), 
      which is equivalent to tf.matmul(input_tensor, w)
  but to perform tf.einsum("BFH,HND->BFND", input_tensor, w),
    call einsum_via_matmul(input_tensor, w, 1)
  tf.einsum("BFND,NDH->BFH", input_tensor, w),
    call einsum_via_matmul(input_tensor, w, 2)
  Args:
    input_tensor: float Tensor of shape [<batch_dims>, <inner_dims>].
    w: float Tensor of shape [<inner_dims>, <outer_dims>].
    num_inner_dims: int. number of dimensions to use for inner products.
  Returns:
    float Tensor of shape [<batch_dims>, <outer_dims>].
  """
  input_shape = get_shape_list(input_tensor)
  w_shape = get_shape_list(w)
  batch_dims = input_shape[: -num_inner_dims]
  inner_dims = input_shape[-num_inner_dims:]
  outer_dims = w_shape[num_inner_dims:]
  inner_dim = np.prod(inner_dims)
  outer_dim = np.prod(outer_dims)
  if num_inner_dims > 1:
    input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
  if len(w_shape) > 2:
    w = tf.reshape(w, [inner_dim, outer_dim])
  ret = tf.matmul(input_tensor, w)
  if len(outer_dims) > 1:
    ret = tf.reshape(ret, batch_dims + outer_dims)
  return ret


def dropout(input_tensor, dropout_prob):
  """Perform dropout.
  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).
  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, rate=dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   head_size,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 3D kernel.
  Args:
    input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
    num_attention_heads: Number of attention heads.
    head_size: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.
  Returns:
    float logits Tensor.
  """

  input_shape = get_shape_list(input_tensor)
  hidden_size = input_shape[2]

  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[hidden_size, num_attention_heads * head_size],
        initializer=initializer)
    w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
    b = tf.get_variable(
        name="bias",
        shape=[num_attention_heads * head_size],
        initializer=tf.zeros_initializer)
    b = tf.reshape(b, [num_attention_heads, head_size])
    ret = tf.einsum("BFH,HND->BFND", input_tensor, w)
    ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        head_size,
                        initializer,
                        activation,
                        name=None):
  """A dense layer with 3D kernel for projection.
  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    hidden_size: The size of hidden layer.
    num_attention_heads: The size of output dimension.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.
  Returns:
    float logits Tensor.
  """
  input_shape = get_shape_list(input_tensor)
  num_attention_heads= input_shape[2]
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[num_attention_heads * head_size, hidden_size],
        initializer=initializer)
    w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
    b = tf.get_variable(
        name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)
    ret = tf.einsum("BFND,NDH->BFH", input_tensor, w)
    ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   num_attention_heads=1,
                   name=None):
  """A dense layer with 2D kernel.
  Args:
    input_tensor: Float tensor with rank 3.
    output_size: The size of output dimension.
    initializer: Kernel initializer.
    activation: Activation function.
    num_attention_heads: number of attention head in attention layer.
    name: The name scope of this layer.
  Returns:
    float logits Tensor.
  """
  del num_attention_heads  # unused
  input_shape = get_shape_list(input_tensor)
  hidden_size = input_shape[2]
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[hidden_size, output_size],
        initializer=initializer)
    b = tf.get_variable(
        name="bias", shape=[output_size], initializer=tf.zeros_initializer)
    ret = tf.einsum("BFH,HO->BFO", input_tensor, w)
    ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dot_product_attention(q, k, v, bias, dropout_rate=0.0):
  """Dot-product attention.
  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
  logits = tf.multiply(logits, 1.0 / math.sqrt(float(get_shape_list(q)[-1])))
  if bias is not None:
    # `attention_mask` = [B, T]
    from_shape = get_shape_list(q)
    if len(from_shape) == 4:
      broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], 1], tf.float32)
    elif len(from_shape) == 5:
      # from_shape = [B, N, Block_num, block_size, depth]#
      broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], from_shape[3],
                                1], tf.float32)

    bias = tf.matmul(broadcast_ones,
                     tf.cast(bias, tf.float32), transpose_b=True)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - bias) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    logits += adder
  else:
    adder = 0.0

  attention_probs = tf.nn.softmax(logits, name="attention_probs")
  attention_probs = dropout(attention_probs, dropout_rate)
  return tf.matmul(attention_probs, v)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.
  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
  size_per_head = int(from_shape[2]/num_attention_heads)

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  # `query_layer` = [B, F, N, H]
  q = dense_layer_3d(from_tensor, num_attention_heads, size_per_head,
                     create_initializer(initializer_range), query_act, "query")

  # `key_layer` = [B, T, N, H]
  k = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                     create_initializer(initializer_range), key_act, "key")
  # `value_layer` = [B, T, N, H]
  v = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                     create_initializer(initializer_range), value_act, "value")
  q = tf.transpose(q, [0, 2, 1, 3])
  k = tf.transpose(k, [0, 2, 1, 3])
  v = tf.transpose(v, [0, 2, 1, 3])
  if attention_mask is not None:
    attention_mask = tf.reshape(
        attention_mask, [batch_size, 1, to_seq_length, 1])
    # 'new_embeddings = [B, N, F, H]'
  new_embeddings = dot_product_attention(q, k, v, attention_mask,
                                         attention_probs_dropout_prob)

  return tf.transpose(new_embeddings, [0, 2, 1, 3])


def attention_ffn_block(layer_input,
                        hidden_size=768,
                        attention_mask=None,
                        num_attention_heads=1,
                        attention_head_size=64,
                        attention_probs_dropout_prob=0.0,
                        intermediate_size=3072,
                        intermediate_act_fn=None,
                        initializer_range=0.02,
                        hidden_dropout_prob=0.0):
  """A network with attention-ffn as sub-block.

  Args:
    layer_input: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    hidden_size: (optional) int, size of hidden layer.
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    attention_head_size: int. Size of attention head.
    attention_probs_dropout_prob: float. dropout probability for attention_layer
    intermediate_size: int. Size of intermediate hidden layer.
    intermediate_act_fn: (optional) Activation function for the intermediate
      layer.
    initializer_range: float. Range of the weight initializer.
    hidden_dropout_prob: (optional) float. Dropout probability of the hidden
      layer.

  Returns:
    layer output
  """

  with tf.variable_scope("attention_1"):
    with tf.variable_scope("self"):
      attention_output = attention_layer(
          from_tensor=layer_input,
          to_tensor=layer_input,
          attention_mask=attention_mask,
          num_attention_heads=num_attention_heads,
          attention_probs_dropout_prob=attention_probs_dropout_prob,
          initializer_range=initializer_range)

    # Run a linear projection of `hidden_size` then add a residual
    # with `layer_input`.
    with tf.variable_scope("output"):
      attention_output = dense_layer_3d_proj(
          attention_output,
          hidden_size,
          attention_head_size,
          create_initializer(initializer_range),
          None,
          name="dense")
      attention_output = dropout(attention_output, hidden_dropout_prob)
  attention_output = layer_norm(attention_output + layer_input)
  with tf.variable_scope("ffn_1"):
    with tf.variable_scope("intermediate"):
      intermediate_output = dense_layer_2d(
          attention_output,
          intermediate_size,
          create_initializer(initializer_range),
          intermediate_act_fn,
          num_attention_heads=num_attention_heads,
          name="dense")
      with tf.variable_scope("output"):
        ffn_output = dense_layer_2d(
            intermediate_output,
            hidden_size,
            create_initializer(initializer_range),
            None,
            num_attention_heads=num_attention_heads,
            name="dense")
      ffn_output = dropout(ffn_output, hidden_dropout_prob)
  ffn_output = layer_norm(ffn_output + attention_output)
  return ffn_output


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_hidden_groups=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      inner_group_num=1,
                      intermediate_act_fn="gelu",
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_hidden_groups: int. Number of group for the hidden layers, parameters
      in the same group are shared.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    inner_group_num: int, number of inner repetition of attention and ffn.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = hidden_size // num_attention_heads
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]

  all_layer_outputs = []
  if input_width != hidden_size:
    prev_output = dense_layer_2d(
        input_tensor, hidden_size, create_initializer(initializer_range),
        None, name="embedding_hidden_mapping_in")
  else:
    prev_output = input_tensor
  with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
    for layer_idx in range(num_hidden_layers):
      group_idx = int(layer_idx / num_hidden_layers * num_hidden_groups)
      with tf.variable_scope("group_%d" % group_idx):
        with tf.name_scope("layer_%d" % layer_idx):
          layer_output = prev_output
          for inner_group_idx in range(inner_group_num):
            with tf.variable_scope("inner_group_%d" % inner_group_idx):
              layer_output = attention_ffn_block(
                  layer_output, hidden_size, attention_mask,
                  num_attention_heads, attention_head_size,
                  attention_probs_dropout_prob, intermediate_size,
                  intermediate_act_fn, initializer_range, hidden_dropout_prob)
              prev_output = layer_output
              all_layer_outputs.append(layer_output)
  if do_return_all_layers:
    return all_layer_outputs
  else:
    return all_layer_outputs[-1]


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.
  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
