import numpy as np
import tensorflow as tf
from typing import Union, Callable, List



def crf(inputs, tag_indices, num_labels, true_sequence_lengths,
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
      initializer=tf.zeros_initializer()
    )
  per_example_loss = None
  if not inference_only:
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
      inputs=inputs,
      tag_indices=tag_indices,
      transition_params=transition_params,
      sequence_lengths=true_sequence_lengths
    )
    per_example_loss = -log_likelihood
  # NOTE CRF decode, decode_tags [batch_size, max_seq_len] most probable path
  decode_tags, best_score = tf.contrib.crf.crf_decode(
    potentials=inputs,
    transition_params=transition_params,
    # NOTE sequence_length: [batch_size] vector of true sequence lengths.
    sequence_length=true_sequence_lengths
  )
  # A [batch_size] Tensor containing the -log_likelihood of each example
  predictions = decode_tags
  return per_example_loss, predictions, best_score


def softmax(logits, labels, num_classes, mask=None):
  """ Perform softmax operation
  Args:
    logits: Logits outputs of the network, last dimension should be num_classes, 
        ie.. [batch_size, seq_length ?, num_classes];
    labels: A [batch_size, seq_length ?] tensor represent true label ids.
    num_classes:
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
  log_probs = tf.nn.log_softmax(logits, axis=-1)  # NOTE same shape as logits
  # NOTE shape (batch_size, max_seq_len, depth)
  per_token_loss = one_hot_labels * log_probs

  if mask is not None and len(per_token_loss.shape.as_list()) > 2:
      mask = tf.cast(mask, tf.float32)
      per_token_loss = tf.einsum("BFH,BF->BFH", per_token_loss, mask)

  per_example_loss = -tf.reduce_sum(per_token_loss, axis=-1)

  probabilities = tf.nn.softmax(logits, axis=-1)  # NOTE same shape as logits
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
    probabilities = tf.nn.softmax(logits, axis=-1)  # NOTE same shape as logits
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int64)
    return per_token_loss, predictions


def contrastive_loss(model1, model2, y, margin):
  with tf.name_scope("contrastive-loss"):
    distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
    similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
    dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
    # give penalty to dissimilar label if the distance is bigger than margin
    return tf.reduce_mean(dissimilarity + similarity) / 2


from tensorflow.python.ops import array_ops


def focal_loss(logits, targets, weights=None, alpha=0.25, gamma=2):
  r"""Compute focal loss for predictions.
      Multi-labels Focal loss formula:
          FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = targets.
  Args:
    logits: A float tensor of shape [batch_size, num_anchors,
      num_classes] representing the predicted logits for each class
    targets: A float tensor of shape [batch_size, num_anchors,
      num_classes] representing one-hot encoded classification targets
    weights: A float tensor of shape [batch_size, num_anchors]
    alpha: A scalar tensor for focal loss alpha hyper-parameter
    gamma: A scalar tensor for focal loss gamma hyper-parameter
  Returns:
      loss: A (scalar) tensor representing the value of the loss function for each samples
  """
  sigmoid_p = tf.nn.sigmoid(logits)
  zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

  # For poitive prediction, only need consider front part loss, back part is 0;
  # targets > zeros <=> z=1, so poitive coefficient = z - p.
  pos_p_sub = array_ops.where(targets > zeros, targets - sigmoid_p, zeros)

  # For negative prediction, only need consider back part loss, front part is 0;
  # targets > zeros <=> z=1, so negative coefficient = 0.
  neg_p_sub = array_ops.where(targets > zeros, zeros, sigmoid_p)
  per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
  return tf.reduce_sum(per_entry_cross_ent, axis=-1)


Number = Union[
  float,
  int,
  np.float16,
  np.float32,
  np.float64,
  np.int8,
  np.int16,
  np.int32,
  np.int64,
  np.uint8,
  np.uint16,
  np.uint32,
  np.uint64,
]

TensorLike = Union[
  List[Union[Number, list]],
  tuple,
  Number,
  np.ndarray,
  tf.Tensor,
  tf.SparseTensor,
  tf.Variable,
]

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


def sigmoid_focal_crossentropy(
        y_true: TensorLike,
        y_pred: TensorLike,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        from_logits: bool = False,
        mask: TensorLike = None,
) -> tf.Tensor:
  """Implements the focal loss function.
  Focal loss was first introduced in the RetinaNet paper
  (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
  classification when you have highly imbalanced classes. It down-weights
  well-classified examples and focuses on hard examples. The loss value is
  much high for a sample which is misclassified by the classifier as compared
  to the loss value corresponding to a well-classified example. One of the
  best use-cases of focal loss is its usage in object detection where the
  imbalance between the background class and other classes is extremely high.
  Args
      y_true: true targets tensor.
      y_pred: predictions tensor.
      alpha: balancing factor.
      gamma: modulating factor.
  Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
      same shape as `y_true`; otherwise, it is scalar.
  """
  import tensorflow.keras.backend as K
  if gamma and gamma < 0:
      raise ValueError("Value of gamma should be greater than or equal to zero")

  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

  # Get the cross_entropy for each entry
  ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

  # If logits are provided then convert the predictions into probabilities
  if from_logits:
      pred_prob = tf.sigmoid(y_pred)
  else:
      pred_prob = y_pred

  p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
  alpha_factor = 1.0
  modulating_factor = 1.0

  if alpha:
      alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
      alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

  if gamma:
      gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
      modulating_factor = tf.pow((1.0 - p_t), gamma)

  # compute the final loss and return
  per_entry_cross_ent = alpha_factor * modulating_factor * ce
  if mask is not None:
      mask = tf.cast(mask, tf.float32)
      per_entry_cross_ent = tf.einsum("BFN,BF->BFN", per_entry_cross_ent, mask)

  return tf.reduce_sum(per_entry_cross_ent, axis=-1)
