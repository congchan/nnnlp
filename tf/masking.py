"""Generate all sort of masks"""

import tensorflow as tf


def upper_left_square_mask(mask):
  """
  Transform a sequence mask to square mask
  :param mask: A mask tensor of shape (batch, feature_size) specify each sample mask
  :return: A mask tensor of shape (batch, feature_size, feature_size)

  Example:
  input
    [ True  True  True  True  True False]
  output
   [[ True  True  True  True  True False]
    [ True  True  True  True  True False]
    [ True  True  True  True  True False]
    [ True  True  True  True  True False]
    [ True  True  True  True  True False]
    [False False False False False False]]]
  """
  return tf.logical_and(
    tf.expand_dims(mask, [1]),  # [batch_size, 1, feature_size]
    tf.expand_dims(mask, [2])  # [batch_size, feature_size, 1]
  )


def upper_triangular_mask(seq_length):
  """
  Return a (seq_length x seq_length) mask which all upper triangular element are true,
    including the diagonal.
  :param seq_length: specify the shape of the matrix
  :return: A mask with shape (seq_length, seq_length)

  '''
  seq_length = 5
  [[ True  True  True  True  True]
   [False  True  True  True  True]
   [False False  True  True  True]
   [False False False  True  True]
   [False False False False  True]]
  '''
  """
  return tf.linalg.band_part(
    tf.ones([seq_length, seq_length], tf.bool), 0, -1,
  )
