import tensorflow as tf



def _luong_score(query, keys, scale):
    """Implements Luong-style (multiplicative) scoring function.
    This attention has two forms.  The first is standard Luong attention,
    as described in:
    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025
    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.
    To enable the second form, call this function with `scale=True`.
    Args:
      query: Tensor, the current target hidden state h_t, 
                shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, each source hidden state \hat(h_s), 
                shape `[batch_size, max_time, num_units]`.
      scale: the optional tensor to scale the attention score.
    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys. "
            "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
            "Perhaps you need to set num_units to the keys' dimension (%s)?"
            % (query, depth, keys, key_units, key_units)
        )

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = tf.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = tf.matmul(query, keys, transpose_b=True)
    score = tf.squeeze(score, [1])

    if scale is not None:
        score = scale * score
    return score


def _bahdanau_score(
    processed_query, keys, attention_v, attention_g=None, attention_b=None
):
    """Implements Bahdanau-style (additive) scoring function.
    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868
    To enable the second form, set please pass in attention_g and attention_b.
    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to
        keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      attention_v: Tensor, shape `[num_units]`.
      attention_g: Optional scalar tensor for normalization.
      attention_b: Optional tensor with shape `[num_units]` for normalization.
    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = (
            attention_g
            * attention_v
            * tf.math.rsqrt(tf.reduce_sum(tf.square(attention_v)))
        )
        return tf.reduce_sum(
            normed_v * tf.tanh(keys + processed_query + attention_b), [2]
        )
    else:
        return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query), [2])