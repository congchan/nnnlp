# coding=utf-8

import os
import logging
from termcolor import colored
from typing import List
import tensorflow as tf
from helper import set_logger, check_tf_version, import_tf
from data import NerProcessor

__all__ = ['__version__']
__version__ = '1.0.0'

_tf_ver_ = check_tf_version()

logger = set_logger(colored('NER', 'yellow'))

class NER(object):
  '''
  NER service
  '''
  def __init__(self, args, graph_path, vocab_file, device_id):
    self.max_seq_length = args.max_seq_length
    self.batch_size = args.batch_size
    self.device_id = device_id
    self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
    self.graph_path = graph_path
    self.gpu_memory_fraction = args.gpu_memory_fraction
    self.data_processor = NerProcessor(vocab_file)
    # load_graph(protobuf_graph_file=protobuf_graph_file)
    tf = import_tf(self.device_id)
    self.estimator = self.get_estimator(tf)
    logger.info('use device %s, load graph from %s' %
              ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))


  def get_estimator(self, tf):
    from tensorflow.python.estimator.estimator import Estimator
    from tensorflow.python.estimator.run_config import RunConfig
    from tensorflow.python.estimator.model_fn import EstimatorSpec

    def model_fn(features, labels, mode, params):
      with tf.gfile.GFile(self.graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

      input_names = ['input_ids', 'input_mask']

      outputs = tf.import_graph_def(
          graph_def,
          input_map={k + ':0': features[k] for k in input_names},
          return_elements=['outputs:0'])

      return EstimatorSpec(
                mode=mode, 
                predictions={
                'outputs': outputs[0]
                }
            )

    config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
    config.log_device_placement = False
    # session-wise XLA doesn't seem to work on tf 1.10
    # if args.xla:
    #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

  def run(self, texts: List[str]):

    for outputs in self.estimator.predict(self.input_fn_builder(tf, texts), yield_single_examples=True):
      yield [r.decode() for r in outputs["outputs"][1: -1]]
        # send_ndarray(sink_embed, r['client_id'], r['encodes'], ServerCmd.data_embed)
        # logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))
    
  def input_fn_builder(self, tf, texts: List[str]):

    def gen():
      examples = self.data_processor.create_examples(texts)
      features = self.data_processor.convert_examples_to_features(examples, self.max_seq_length)
      for feature in features:
        yield {
            'input_ids': feature.input_ids,
            'input_mask': feature.input_mask
        }

    def input_fn():
      return (tf.data.Dataset.from_generator(
          gen,
          output_types={'input_ids': tf.int32,
                        'input_mask': tf.int32,
                        },
          output_shapes={
              'input_ids': (None),
              'input_mask': (None),
              }).batch(self.batch_size).prefetch(self.prefetch_size))

    return input_fn
