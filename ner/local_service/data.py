# coding=utf-8
# Copyright 2020 The Pingannlp Team Authors.

from typing import List
from tokenization import load_vocab, convert_to_unicode, convert_by_vocab

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, guid, tokens, input_ids, input_mask):
    self.guid = guid
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask


class InputExample(object):
  """A single test example for sequence ner."""

  def __init__(self, guid, text, tokens):
    """Constructs a InputExample. contains text

    Args:
      guid: Unique id for the example.
      text: the original text.
      entity: (Optional), The entity label sequence of the example. 
          This should be specified for train and dev examples.
    """
    self.guid = guid
    self.text = text
    self.tokens = tokens


class NerProcessor(object):
  """ Processor for the NER task.  """
  
  def __init__(self, vocab_file, do_lower_case=True, unk_token="[UNK]"):
    self.tokenizer = Tokenizer(vocab_file=vocab_file, 
                               do_lower_case=do_lower_case,
                               unk_token=unk_token)

  def get_labels(self):
    """ 'O' for OTHERS, start position and padding,
        should be the first to get a lable idx of 0. 
    """
    return ['O',
            'B-ZIP', 'I-ZIP', 'B-TIME', 'I-TIME', 'B-NUM', 'I-NUM', # 6
            'B-COM', 'I-COM', 'B-SCH', 'I-SCH', 'B-ORG', 'I-ORG',   # 6
            'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC',                     # 4
            'B-TITLE', 'I-TITLE', 'B-PER', 'I-PER',                 # 4
            'B-EDU', 'I-EDU', 'B-PRO', 'I-PRO', 'B-WORK', 'I-WORK'  # 6
    ]

  def create_examples(self, texts: List[str]):
    examples = []
    for (i, text) in enumerate(texts):
      if len(text) == 0:
        continue
      guid = "%s-%s" % ("predict", i)
      examples.append(InputExample(guid=guid, text=text, 
          tokens=self.tokenizer.tokenize(text)))
    return examples


  def convert_examples_to_features(self, examples, max_seq_length=128, 
                                  ):
    """Convert a set of `InputExample`s into a list of `InputBatch`s."""

    def convert_single_example(example):
      """Converts a single `InputExample` into one or more `InputFeatures`.
        text will be split if the sequence length exceed the max_seq_length.
        return a feature list
      """
      n_token_holders = 2
      tokens = ['CLS'] + example.tokens[:max_seq_length-n_token_holders] + ['SEP']
      input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length. more pythonic
      pad_len = max_seq_length - len(input_ids)
      input_ids += [0] * pad_len
      input_mask += [0] * pad_len

      try:
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        yield InputFeatures(
            guid=example.guid,
            tokens=example.tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            )
      except:
        print("Error: ", example.guid, example.text)
        
    for example in examples:
      yield from convert_single_example(example=example)



class Tokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True, unk_token="[UNK]"):
    self.unk_token = unk_token
    self.do_lower_case = do_lower_case
    self.vocab = load_vocab(vocab_file)
    self.unk_token_id = self.vocab[self.unk_token]
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text):
    text = convert_to_unicode(text)
    return list(text.lower())

  def convert_tokens_to_ids(self, tokens):
    return [self.vocab.get(token.lower(), self.unk_token_id) for token in tokens]

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)

