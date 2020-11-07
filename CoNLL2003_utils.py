# coding=utf-8
"""Utility functions for CoNLL-2003 Ner tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tokenization


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

  def get_labels(self):
    """
    "X" represents "##eer","##soo" and so on!
    "O" represents 'OTHERS' entity and padding
    :return:
    """
    return ["O","B-MISC", "I-MISC", "B-PER", "I-PER", 
        "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X"]
