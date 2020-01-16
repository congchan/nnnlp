" Tensorflow version 1.x modeling codes. "
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import math
import six
from six.moves import range
import tensorflow as tf


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

  if mask is not None:
    mask = tf.cast(mask, tf.float32)
    per_token_loss = tf.einsum("BFH,BF->BFH", per_token_loss, mask)

  per_example_loss = -tf.reduce_sum(per_token_loss, axis=-1)

  probabilities = tf.nn.softmax(logits, axis=-1) # NOTE same shape as logits
  predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int64)

  return per_example_loss, predictions


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


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

