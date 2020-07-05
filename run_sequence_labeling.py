# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT NER trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from bert import modeling
from bert import optimization
import tokenization
import tensorflow as tf
import numpy as np
import json
import itertools
import logging
from util import *
from modeling import *
from CoNLL2003_utils import *

flags = tf.flags

FLAGS = flags.FLAGS

## NOTE self defined parameter
flags.DEFINE_string( "load_config_file", None, " if specified, load config from file.")
flags.DEFINE_integer( "sentence_overlap", 16,
    "When splitting up a long sentence into chunks, how much overlap to "
    "take between chunks.")
flags.DEFINE_integer('keep_checkpoint_max', 5, 'max num of checkpoints to keep.')
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_float("bilstm_dropout_rate", 0.2, "Proportion of dropout for bilstm.")
flags.DEFINE_bool("use_bilstm", False, "Whether to use bilstm for decoding.")
flags.DEFINE_bool("use_crf", False, "Whether to use crf for decoding.")
flags.DEFINE_bool("data_augmentation", False, "Whether to use data augmentation.")
flags.DEFINE_float("debug", float('inf'), "Debug on small data set.")
flags.DEFINE_integer('shard_size', 8192, 'size of eac tf record shard')
flags.DEFINE_string( "init_checkpoint_dir", './chinese_L-12_H-768_A-12',
    "Initial checkpoint dir "
    "Which contains a pre-trained BERT model, config file, vocab file.")
flags.DEFINE_string( "finetune_checkpoint", None,
    "Checkpoint to be finetuned")
flags.DEFINE_string( "train_file", None, "The file contain train data.")    
flags.DEFINE_string( "eval_file", None, "The file contain eval data.")
flags.DEFINE_string( "test_file", None, "The file contain test data.")
flags.DEFINE_string( "pred_results_file", 'pred_results.json', "The file contain predicted results.")
flags.DEFINE_string( "checkpoint_path", None,
    "Path of a specific checkpoint to predict. "
    "If None, the latest checkpoint in model_dir is used.")
flags.DEFINE_bool( "read_record", False, 
    "Whether to read the training data from output_dir"
    "The output_dir should contain train.tf_record file.")

## Required parameters
flags.DEFINE_string("task_name", 'ner', "The name of the task to train.")

flags.DEFINE_string(
    "output_dir", './output',
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 512, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 512, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

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


def file_based_convert_examples_to_features(
    examples, label_map, max_seq_length, seq_overlap, tokenizer, output_files):
  """Convert a set of `InputExample`s to TFRecord files."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0
  total_written = 0
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    for feature in convert_single_example(ex_index=ex_index, example=example,
                                         label_map=label_map,
                                         tokenizer=tokenizer,
                                         max_seq_length=max_seq_length,
                                         ):
      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["label_ids"] = create_int_feature(feature.label_ids)
      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      # writer.write(tf_example.SerializeToString())
      writers[writer_index].write(tf_example.SerializeToString())
      writer_index = (writer_index + 1) % len(writers)
      total_written += 1

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def file_based_input_fn_builder(input_files, seq_length, is_training,
                                drop_remainder, num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if FLAGS.use_tpu and t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      # shuffle files list
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=2048)
    else:
      d = tf.data.TFRecordDataset(input_files)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, 
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a ner model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  bert_output = model.get_sequence_output()
  # NOTE bert_output: [batch_size, seq_length, hidden_size]
  hidden_size = bert_output.shape[-1].value  
  if is_training: # I.e., 0.1 dropout
    bert_output = tf.nn.dropout(bert_output, keep_prob=0.9)
  output_layer = bert_output
  _true_length = tf.cast(tf.reduce_sum(input_mask, axis=-1), dtype=tf.int32)

  # NOTE here are the customized layers
  if FLAGS.use_bilstm:
    with tf.variable_scope("lstm"):
      output_layer = bilstm_fused(output_layer, _true_length, FLAGS.lstm_size,
                              FLAGS.bilstm_dropout_rate, is_training)
    hidden_size = output_layer.shape[-1].value 

  # NOTE For your specific task purpose
  with tf.variable_scope("decoder"):
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
    # NOTE num_labels; 
    # logits: Tensor(shape=(batch_size, max_seq_len, num_labels), 
    # labels: Tensor(shape=(batch_size, max_seq_len)
    tf.logging.info("num_labels: {}; logits: {}; labels:{}".format(
        num_labels, logits, labels))

    if FLAGS.use_crf:
      per_example_loss, predictions, _ = crf_layer(inputs=logits, 
          tag_indices=labels, 
          num_labels=num_labels, 
          true_sequence_lengths=_true_length, 
          transitions_name="transitions", 
          )
    else:
      per_example_loss, predictions = softmax(logits=logits,
          labels=labels, num_classes=num_labels, mask=input_mask)
    
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, predictions)


def model_fn_builder(bert_config, label_list, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  num_labels = len(label_list)

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, predictions) = create_model(
        bert_config, is_training, input_ids, input_mask, label_ids,
        num_labels, use_one_hot_embeddings)

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

    global_step = tf.train.get_or_create_global_step()
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      logging_hook = tf.train.LoggingTensorHook({"Training loss": total_loss,
          "Global step": global_step},  
          every_n_iter=500)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      
      def metric_fn(per_example_loss, label_ids, logits, input_mask):
        # NOTE logits (?, max_seq_len, num_labels), label_ids (?, max_seq_len)
        print("###metric_fn:", logits.shape, label_ids.shape)
        weighted_loss = tf.metrics.mean(values=per_example_loss, 
            weights=tf.reduce_sum(input_mask, axis=-1) if FLAGS.use_crf else input_mask,)        
        # confusion_matrix, cm_op = UnbalancedMetrics.streaming_confusion_matrix(
        #     labels=label_ids, 
        #     predictions=predictions, 
        #     num_classes=num_labels, 
        #     weights=input_mask)
        return {
            "weighted_loss": weighted_loss
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, input_mask])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      # label_table = tf.contrib.lookup.index_to_string_table_from_tensor(
      #     tf.constant(label_list), default_value=label_list[0])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"predictions": tf.nn.embedding_lookup(
                                          tf.constant(label_list), 
                                          predictions),
                       #label_table.lookup(predictions), 
                       },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn



###############################################################################
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "ner": CoNLL2013Processor,
  }

  skip_flags = ('load_config_file', 'do_train', 'do_eval', 'do_predict',
                'eval_fole', 'test_file', 'pred_results_file', 'checkpoint_path')
  if FLAGS.load_config_file:
    with open(FLAGS.load_config_file, 'r') as reader:
      for name, value in json.loads(reader.read()).items():
        if name not in skip_flags:
          FLAGS.__flags[name].value = value

  bert_config_file = os.path.join(FLAGS.init_checkpoint_dir, 'bert_config.json')
  vocab_file = os.path.join(FLAGS.init_checkpoint_dir, 'vocab.txt')
  if FLAGS.finetune_checkpoint:
    init_checkpoint = FLAGS.finetune_checkpoint
  else:
    init_checkpoint = os.path.join(FLAGS.init_checkpoint_dir, 'bert_model.ckpt')
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  # NOTE get TF logger
  logging.basicConfig(
      level=logging.INFO,
      format="INFO#%(asctime)s# %(message)s",
      datefmt='%Y-%m-%d %H:%M:%S',
      handlers=[logging.FileHandler(os.path.join(FLAGS.output_dir, 'log.log'))]
  )
  tf.logging.info("*"*10+" Config "+"*"*10+": \n{}".format(
      FLAGS.flag_values_dict()))
  if FLAGS.do_train:
    with open(os.path.join(FLAGS.output_dir, 'config.json'), 
              'w', encoding='utf-8') as writer:
      writer.write(to_json_string(FLAGS.flag_values_dict()))

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      label_list=label_list,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    num_train_files = -(-len(train_examples) // FLAGS.shard_size)
    train_files = [os.path.join(FLAGS.output_dir, "train.tf_record_"+f"{i:05d}") 
                    for i in range(num_train_files)]
    tf.logging.info("*** Writing training examples to tf record files ***")
    for train_file in train_files:
      tf.logging.info("  %s", train_file)                    

    if not FLAGS.read_record:
      [tf.gfile.Remove(_f) for _f in tf.gfile.Glob(
          os.path.join(FLAGS.output_dir, "train.tf_record*"))]
      file_based_convert_examples_to_features(
        train_examples, label_map, FLAGS.max_seq_length, FLAGS.seq_overlap, tokenizer, train_files)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_files=train_files,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

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
        eval_examples.append(PaddingInputExample())

    eval_files = [os.path.join(FLAGS.output_dir, "eval.tf_record")]
    file_based_convert_examples_to_features(
        eval_examples, label_map, FLAGS.max_seq_length, FLAGS.seq_overlap, tokenizer, eval_files)

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
    eval_input_fn = file_based_input_fn_builder(
        input_files=eval_files,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    tf.logging.info("***** Eval results *****")
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    results = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, 
        checkpoint_path=FLAGS.checkpoint_path)
    with open(output_eval_file, "a") as writer:
      tf.logging.info(write_eval(writer, results, label_list, label_list[1:]))

    # NOTE self-defined metric
    predictions = estimator.predict(input_fn=eval_input_fn, 
        checkpoint_path=FLAGS.checkpoint_path,
        yield_single_examples=True)
    predict_entitys = reconstruct_from_estimator(eval_examples, predictions)
    try:
        next(predictions)
    except StopIteration:
        print("Complete Evaluation!")
    else:
      tf.logging.info("ERROR: Output examples number not matched! \
          This is likely due to bugs in splitting and reconstruct long text ")

    pred = [[e.decode() for e in entity] for entity in predict_entitys]
    label = [example.text_entity for example in eval_examples]
    sb_measure = SpanBasedF1Measure()
    sb_measure(pred, label)
    tf.logging.info(sb_measure.log_measure(output_eval_file))

    
  if FLAGS.do_predict:
    " For demo only, the actual predict phase should be run by other API "
    predict_examples = processor.get_test_examples(FLAGS.data_dir)

    predict_files = [os.path.join(FLAGS.output_dir, "predict.tf_record")]
    file_based_convert_examples_to_features(predict_examples, label_map,
                                            FLAGS.max_seq_length, FLAGS.seq_overlap,
                                            tokenizer, predict_files)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_files=predict_files,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    predict_results = estimator.predict(input_fn=predict_input_fn, 
        checkpoint_path=FLAGS.checkpoint_path,
        yield_single_examples=True)

    output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.pred_results_file)
    tf.logging.info("***** Predict results *****")
    # Writing prediction results for CoNLL-2003 perl evaluation conlleval.pl
    with open(output_predict_file, 'w', encoding='utf-8') as writer:
      for example, predict_result in zip(predict_examples, predict_results):
        tokens = [token for token in tokenizer.tokenize(word) 
                            for word in example.words]
        words = (word for word in example.words)
        for i, tag in enumerate(predict_result):
          if tokens[i].startswith("##"):
            continue
          line = "{}\t{}\t{}\n".format(next(words), example.labels[i], tag)
          writer.write(line)


if __name__ == "__main__":
  tf.app.run()
