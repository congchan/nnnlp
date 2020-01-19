# coding=utf-8
"""GLUE tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import six
import json
import copy
import glue_utils as classifier_utils
import modeling
import optimization
import tokenization
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import metrics as contrib_metrics

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "config_file", None,
    "The config json file corresponding to the pre-trained model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("cached_dir", None,
                    "Path to cached training and dev tfrecord file. "
                    "The file will be generated if not exist.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("train_step", 1000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "warmup_step", 0,
    "number of steps to perform linear learning rate warmup for.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("optimizer", "adamw", "Optimizer to use")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class Config(object):
  """Configuration."""

  def __init__(self,
               vocab_size,
               embedding_size=128,
               hidden_size=4096,
               num_hidden_layers=12,               
               num_hidden_groups=1,
               num_attention_heads=64,
               intermediate_size=16384,
               inner_group_num=1,
               down_scale_factor=1,
               hidden_act="gelu",
               hidden_dropout_prob=0,
               attention_probs_dropout_prob=0,
               max_position_embeddings=512,
               type_vocab_size=2,
               initializer_range=0.02,
               num_bilstm=1,
               lstm_size=128,
               bilstm_dropout_rate=0.2):
    """Constructs Config.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `Model`.
      embedding_size: size of voc embeddings.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_hidden_groups: Number of group for the hidden layers, parameters in
        the same group are shared.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      inner_group_num: int, number of inner repetition of attention and ffn.
      down_scale_factor: float, the scale to apply
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `Model`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      num_bilstm: The number of bilstm layer.
      lstm_size: The hidden size of bilstm state.
      bilstm_dropout_rate: The dropout rate of bilstm.
    """
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_groups = num_hidden_groups
    self.num_attention_heads = num_attention_heads
    self.inner_group_num = inner_group_num
    self.down_scale_factor = down_scale_factor
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.num_bilstm=num_bilstm
    self.lstm_size=lstm_size
    self.bilstm_dropout_rate=bilstm_dropout_rate

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `Config` from a Python dictionary of parameters."""
    config = Config(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `Config` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def create_model(config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, task_name,):
  """Creates a classification model from_scratch."""
  _true_length = tf.cast(tf.reduce_sum(input_mask, axis=-1), dtype=tf.int32)

  with tf.variable_scope("baseline"):
    with tf.variable_scope("embeddings"):
      # Perform embedding lookup on the word ids.
      (word_embedding_output,
        output_embedding_table) = modeling.embedding_lookup(
          input_ids=input_ids,
          vocab_size=config.vocab_size,
          embedding_size=config.embedding_size,
          initializer_range=config.initializer_range,
          word_embedding_name="word_embeddings",
          use_one_hot_embeddings=use_one_hot_embeddings)

      # Add positional embeddings and token type embeddings, then layer
      # normalize and perform dropout.
      embedding_output = modeling.embedding_postprocessor(
          input_tensor=word_embedding_output,
          use_token_type=True,
          token_type_ids=segment_ids,
          token_type_vocab_size=config.type_vocab_size,
          token_type_embedding_name="token_type_embeddings",
          use_position_embeddings=True,
          position_embedding_name="position_embeddings",
          initializer_range=config.initializer_range,
          max_position_embeddings=config.max_position_embeddings,
          dropout_prob=config.hidden_dropout_prob)
  with tf.variable_scope("bilstm"):
    sequence_output = modeling.bilstm_fused(
      inputs=embedding_output, 
      lengths=_true_length, 
      lstm_size=config.lstm_size,
      bilstm_dropout_rate=config.bilstm_dropout_rate, 
      is_training=is_training,
      num_layers=config.num_bilstm)

  # first_token_tensor = tf.squeeze(sequence_output[:, -1:, :], axis=1)
  last_token_tensor = tf.squeeze(sequence_output[:, -1:, :], axis=1)
  output_layer = tf.layers.dense(
            last_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(config.initializer_range))

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if task_name != "sts-b":
      probabilities = tf.nn.softmax(logits, axis=-1)
      predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    else:
      probabilities = logits
      logits = tf.squeeze(logits, [-1])
      predictions = logits
      per_example_loss = tf.square(logits - labels)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, probabilities, logits, predictions)


def model_fn_builder(config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name,
                     optimizer="adamw"):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, probabilities, logits, predictions) = \
        create_model(config, is_training, input_ids, input_mask,
                     segment_ids, label_ids, num_labels,
                     use_one_hot_embeddings, task_name)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      if task_name not in ["sts-b", "cola"]:
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          loss = tf.metrics.mean(
              values=per_example_loss, weights=is_real_example)
          return {
              "eval_accuracy": accuracy,
              "eval_loss": loss,
          }
      elif task_name == "sts-b":
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          """Compute Pearson correlations for STS-B."""
          # Display labels and predictions
          concat1 = contrib_metrics.streaming_concat(logits)
          concat2 = contrib_metrics.streaming_concat(label_ids)

          # Compute Pearson correlation
          pearson = contrib_metrics.streaming_pearson_correlation(
              logits, label_ids, weights=is_real_example)

          # Compute MSE
          # mse = tf.metrics.mean(per_example_loss)
          mse = tf.metrics.mean_squared_error(
              label_ids, logits, weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          return {"pred": concat1, "label_ids": concat2, "pearson": pearson,
                  "MSE": mse, "eval_loss": loss,}
      elif task_name == "cola":
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          """Compute Matthew's correlations for STS-B."""
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
          tp, tp_op = tf.metrics.true_positives(
              predictions, label_ids, weights=is_real_example)
          tn, tn_op = tf.metrics.true_negatives(
              predictions, label_ids, weights=is_real_example)
          fp, fp_op = tf.metrics.false_positives(
              predictions, label_ids, weights=is_real_example)
          fn, fn_op = tf.metrics.false_negatives(
              predictions, label_ids, weights=is_real_example)

          # Compute Matthew's correlation
          mcc = tf.div_no_nan(
              tp * tn - fp * fn,
              tf.pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5))

          # Compute accuracy
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          return {"matthew_corr": (mcc, tf.group(tp_op, tn_op, fp_op, fn_op)),
                  "eval_accuracy": accuracy, "eval_loss": loss,}

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "probabilities": probabilities,
              "predictions": predictions
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": classifier_utils.ColaProcessor,
      "mnli": classifier_utils.MnliProcessor,
      "mismnli": classifier_utils.MisMnliProcessor,
      "mrpc": classifier_utils.MrpcProcessor,
      "rte": classifier_utils.RteProcessor,
      "sst-2": classifier_utils.Sst2Processor,
      "sts-b": classifier_utils.StsbProcessor,
      "qqp": classifier_utils.QqpProcessor,
      "qnli": classifier_utils.QnliProcessor,
      "wnli": classifier_utils.WnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  if FLAGS.config_file:
    config = Config.from_json_file(
        FLAGS.config_file)
    if FLAGS.max_seq_length > config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the model "
          "was only trained up to sequence length %d" %
          (FLAGS.max_seq_length, config.max_position_embeddings))
  else:
    config = None  # Get the config from TF-Hub.

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name](
      do_lower_case=FLAGS.do_lower_case)

  label_list = processor.get_labels()
  
  tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  if FLAGS.do_train:
    iterations_per_loop = int(min(FLAGS.iterations_per_loop,
                                  FLAGS.save_checkpoints_steps))
  else:
    iterations_per_loop = FLAGS.iterations_per_loop
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=int(FLAGS.save_checkpoints_steps),
      keep_checkpoint_max=0,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
  model_fn = model_fn_builder(
      config=config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.train_step,
      num_warmup_steps=FLAGS.warmup_step,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      task_name=task_name,
      optimizer=FLAGS.optimizer)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    cached_dir = FLAGS.cached_dir
    if not cached_dir:
      cached_dir = FLAGS.output_dir
    train_file = os.path.join(cached_dir, task_name + "_train.tf_record")
    if not tf.gfile.Exists(train_file):
      classifier_utils.file_based_convert_examples_to_features(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer,
          train_file, task_name)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", FLAGS.train_step)
    train_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        task_name=task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_step)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(classifier_utils.PaddingInputExample())

    cached_dir = FLAGS.cached_dir
    if not cached_dir:
      cached_dir = FLAGS.output_dir
    eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")
    if not tf.gfile.Exists(eval_file):
      classifier_utils.file_based_convert_examples_to_features(
          eval_examples, label_list, FLAGS.max_seq_length, tokenizer,
          eval_file, task_name)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        task_name=task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.eval_batch_size)

    best_trial_info_file = os.path.join(FLAGS.output_dir, "best_trial.txt")

    def _best_trial_info():
      """Returns information about which checkpoints have been evaled so far."""
      if tf.gfile.Exists(best_trial_info_file):
        with tf.gfile.GFile(best_trial_info_file, "r") as best_info:
          global_step, best_metric_global_step, metric_value = (
              best_info.read().split(":"))
          global_step = int(global_step)
          best_metric_global_step = int(best_metric_global_step)
          metric_value = float(metric_value)
      else:
        metric_value = -1
        best_metric_global_step = -1
        global_step = -1
      tf.logging.info(
          "Best trial info: Step: %s, Best Value Step: %s, "
          "Best Value: %s", global_step, best_metric_global_step, metric_value)
      return global_step, best_metric_global_step, metric_value

    def _remove_checkpoint(checkpoint_path):
      for ext in ["meta", "data-00000-of-00001", "index"]:
        src_ckpt = checkpoint_path + ".{}".format(ext)
        tf.logging.info("removing {}".format(src_ckpt))
        tf.gfile.Remove(src_ckpt)

    def _find_valid_cands(curr_step):
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      candidates = []
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          idx = ckpt_name.split("-")[-1]
          if int(idx) > curr_step:
            candidates.append(filename)
      return candidates

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

    if task_name == "sts-b":
      key_name = "pearson"
    elif task_name == "cola":
      key_name = "matthew_corr"
    else:
      key_name = "eval_accuracy"

    global_step, best_perf_global_step, best_perf = _best_trial_info()
    writer = tf.gfile.GFile(output_eval_file, "w")
    while global_step < FLAGS.train_step:
      steps_and_files = {}
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
          gstep = int(cur_filename.split("-")[-1])
          if gstep not in steps_and_files:
            tf.logging.info("Add {} to eval list.".format(cur_filename))
            steps_and_files[gstep] = cur_filename
      tf.logging.info("found {} files.".format(len(steps_and_files)))
      if not steps_and_files:
        tf.logging.info("found 0 file, global step: {}. Sleeping."
                        .format(global_step))
        time.sleep(60)
      else:
        for checkpoint in sorted(steps_and_files.items()):
          step, checkpoint_path = checkpoint
          if global_step >= step:
            if (best_perf_global_step != step and
                len(_find_valid_cands(step)) > 1):
              _remove_checkpoint(checkpoint_path)
            continue
          result = estimator.evaluate(
              input_fn=eval_input_fn,
              steps=eval_steps,
              checkpoint_path=checkpoint_path)
          global_step = result["global_step"]
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
          writer.write("best = {}\n".format(best_perf))
          if result[key_name] > best_perf:
            best_perf = result[key_name]
            best_perf_global_step = global_step
          elif len(_find_valid_cands(global_step)) > 1:
            _remove_checkpoint(checkpoint_path)
          writer.write("=" * 50 + "\n")
          writer.flush()
          with tf.gfile.GFile(best_trial_info_file, "w") as best_info:
            best_info.write("{}:{}:{}".format(
                global_step, best_perf_global_step, best_perf))
    writer.close()

    for ext in ["meta", "data-00000-of-00001", "index"]:
      src_ckpt = "model.ckpt-{}.{}".format(best_perf_global_step, ext)
      tgt_ckpt = "model.ckpt-best.{}".format(ext)
      tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
      tf.io.gfile.rename(
          os.path.join(FLAGS.output_dir, src_ckpt),
          os.path.join(FLAGS.output_dir, tgt_ckpt),
          overwrite=True)

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(classifier_utils.PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    classifier_utils.file_based_convert_examples_to_features(
        predict_examples, label_list,
        FLAGS.max_seq_length, tokenizer,
        predict_file, task_name)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        task_name=task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.predict_batch_size)

    checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
    result = estimator.predict(
        input_fn=predict_input_fn,
        checkpoint_path=checkpoint_path)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    output_submit_file = os.path.join(FLAGS.output_dir, "submit_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as pred_writer,\
        tf.gfile.GFile(output_submit_file, "w") as sub_writer:
      sub_writer.write("index" + "\t" + "prediction\n")
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, (example, prediction)) in\
          enumerate(zip(predict_examples, result)):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        pred_writer.write(output_line)

        if task_name != "sts-b":
          actual_label = label_list[int(prediction["predictions"])]
        else:
          actual_label = str(prediction["predictions"])
        sub_writer.write(example.guid + "\t" + actual_label + "\n")
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
