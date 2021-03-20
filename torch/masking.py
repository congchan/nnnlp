"""Generate all sort of masks"""

import torch


def sequence_mask(lengths, max_seq_length=None, dtype=torch.bool):
  """
  Returns a mask tensor representing the first N positions of each cell.

  lengths:	integer tensor, all its values <= maxlen.
  max_seq_length:	scalar integer tensor, size of last dimension of returned tensor.
    Default is the maximum value in lengths.
  dtype:	output type of the resulting tensor.
  return: A mask tensor of shape lengths.shape + (maxlen,), cast to specified dtype.

  sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                #  [True, True, True, False, False],
                                #  [True, True, False, False, False]]

  sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                  #   [True, True, True]],
                                  #  [[True, True, False],
                                  #   [False, False, False]]]
  """
  if max_seq_length is None:
    max_seq_length = lengths.max()
  matrix = torch.unsqueeze(lengths, dim=-1)
  row_vector = torch.arange(0, max_seq_length, 1).to(matrix.device)
  mask = row_vector < matrix

  mask.type(dtype)
  return mask


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
  return torch.logical_and(
    mask.unsqueeze(1),  # [batch_size, 1, feature_size]
    mask.unsqueeze(2)  # [batch_size, feature_size, 1]
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
  return torch.logical_and(
    seq_length.unsqueeze(1),  # [batch_size, 1, seq_length]
    seq_length.unsqueeze(2)  # [batch_size, seq_length, 1]
  )
