# coding=utf-8
"""Utility functions for CoNLL-2003 Ner tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tokenization


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
  """A single test example for sequence labeling data."""

  def __init__(self, guid, words, labels=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      words: list of str. The untokenized text of the sequence. For single
        sequence tasks, only this sequence must be specified.
      labels: (Optional) list of str. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.words = words
    self.labels = labels


class CoNLL2013Processor():
  @classmethod
  def _read_bio(cls, input_file):
    """ Read sequence data from BIO schema data.
    return list of data as format {'tokens': tokens, 'tags': tags} 
    """
    rf = open(input_file,'r')
    lines = []; words = []; tags = []
    for line in rf:
      word = line.strip().split(' ')[0]
      tag = line.strip().split(' ')[-1]
      if len(line.strip())==0 and words[-1] == '.':
          words = [word for word in words if len(word) > 0]
          tags = [tag for tag in tags if len(tag) > 0]
          lines.append({'words': words, 'tags': tags})
          words=[]
          tags = []
      words.append(word)
      tags.append(tag)
    rf.close()
    return lines

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_bio(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_bio(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_bio(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_labels(self):
        """
        "O" represents 'OTHERS' entity and padding
        :return:
        """
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", 
            "B-ORG", "I-ORG", "B-LOC", "I-LOC"]    

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            words = tokenization.convert_to_unicode(line['words'])
            labels = tokenization.convert_to_unicode(line['tags'])
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, guid, input_ids, input_mask,
               label_ids=None, tokens=None):
    self.guid = guid
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.label_ids = label_ids
    self.tokens = tokens


def convert_single_example(ex_index, example, label_map, tokenizer, 
    max_seq_length=128,):
  """Converts a single `InputExample` into one or more `InputFeatures`.
     text will be split if the sequence length exceed the max_seq_length.

    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    example:[      is this, jack,  ##son?]
    labels: [[PAD] O, O,    B-PER, I-PER]

    return a feature list
  """
  input_ids = tokenizer.convert_tokens_to_ids(example.tokens[:max_seq_length])
  label_ids = label_map[example.labels]
  labels = example.labels

  tokens = ['O',]
  labels = ['O',]
  for i, (word, label) in enumerate(zip(example.words, example.labels)):
    tokened_word = tokenizer.tokenize(word)
    tokens.extend(tokened_word)
    for i, _ in enumerate(tokened_word):
      if i==0:
        labels.append(label)
      else:
        labels.append(label.replace('B-', 'I-'))
  # NOTE here we change the notation in BERT
  # (b) For single sequences:
  #  tokens:    is this jackson ? 
  #  labels:  O O  O    B-      O O O ...
  #  mask:    1 1  1    1       1 0 0 ...
  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])
  label_ids = [label_map[label] for label in labels[:max_seq_length]]
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length. more pythonic
  pad_len = max_seq_length - len(input_ids)
  input_ids += [0] * pad_len
  input_mask += [0] * pad_len

  try:
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    if ex_index < 3:
      logging.info("*** Example ***")
      logging.info("guid: %s" % (example.guid))
      logging.info("type: %s" % (example.set_type))
      logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logging.info("label: {} (id = {})".format(labels, label_ids))

    yield InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        label_ids=label_ids)
  except:
    print(example.guid, example.words)


def convert_examples_to_features(examples, label_map, tokenizer, 
    max_seq_length=128):
  """Convert a set of `InputExample`s into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tmp_feature = convert_single_example(ex_index=ex_index, example=example,
                                         label_map=label_map,
                                         tokenizer=tokenizer,
                                         max_seq_length=max_seq_length,
                                         )
    features.extend(list(tmp_feature))
  return features
