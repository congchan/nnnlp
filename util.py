# coding=utf-8
import os, json, csv, time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys
import re
import unittest
import copy


__all__ = ['read_json_list', 'write_file', 'write_json', 'read_text',
           'UnbalancedMetrics', 'write_eval', 'write_confusion_matrix',
           'SpanBasedF1Measure', 
           'slide_over_sequence','split_paragraph',
           'reconstruct_entity', 'reconstruct_from_estimator',
           'to_json_string',
           'get_entity_from_bio',
          ]

def read_json_list(path, debug=None):
    " load a list of dict from json file "
    data = []
    line_num = 1
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if debug and debug < line_num:
                break
            line_num += 1
            data.append(json.loads(line))
    return data

def read_text(path, debug=None):
    " read in text data "
    data = []
    line_num = 1
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            if debug and debug < line_num:
                break
            line_num += 1
            data.append(line.strip())
    return data

def write_file(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line+'\n')


def write_json(data, path):
    " write json data to file"
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False))
            f.write('\n')



def get_entity_from_bio(tokens, entitys):
    """
    Get {word:entity} from bio labeled sequence,. 
    
    Only accept BIO schema.
    inputs:
    tokens:     A     B     C     D     E F
    entitys:    B-GPE I-GPE B-PER I-PER O O
    return: {'AB': 'GPE', 'CD': 'PER'}
    """
    word2entity = {}
    left = 0
    right = 1

    def _forward(idx, mark):
        while (idx < len(entitys) and entitys[idx][0] == mark):
            idx += 1
        return idx

    def _loc(idx, mark):
        while (idx < len(entitys) and entitys[idx][0] != mark):
            idx += 1
        return idx

    def _get_next(idx, marks):
        while (idx < len(entitys) and entitys[idx][0] not in marks):
            idx += 1
        return idx

    def _add(left, right):
        word = ''.join(tokens[left: right])
        entity = list(set([entity[2:] for entity in entitys[left: right]]))
        assert len(entity) == 1
        word2entity[word] = entity[0]

    while left < len(entitys):
        if entitys[left][0] == 'B':
            if right < len(entitys) and entitys[right][0] == 'I':
                right = _get_next(right, ('B', 'O'))
            _add(left, right)
            left = right
            right += 1
        else:
            left = _get_next(left, ('B'))
            right = left + 1
    
    return word2entity


def split_paragraph(sentence_tokens, stop_signals=['。', '！', '？', '，', '、',], 
            window_size=128, seq_overlap=24):
    """
    For long paragraph, we need to separate it into several sentence(w/o overlap).
    Split the sentence with slide windows to maintain some context(seq_overlap). 
    :param sentence_tokens:
    :param stop_signals:
    :param window_size:
    :param seq_overlap:
    :return:
    sequence_spans: List[left: int, right: int, start: int, end: int]
        left, right: the position span in the original sequence 
        start, end: the actual sequence span we care about (ingore overlap)
    """
    sequence_spans = slide_over_sequence(sentence_tokens, stop_signals=stop_signals,
            window_size=window_size, seq_overlap=seq_overlap)
    return sequence_spans

def split_paragraph_2(sentence_tokens, stop_signals=['。', '！', '？', '，', '、',], 
            window_size=127, seq_overlap=16):
    """
    For long paragraph, we need to separate it into several sentence(w/o overlap).
    Split the sentence with slide windows to maintain some context(seq_overlap). 
    """
    if len(text) > window_size: 
        paragraph_spans = []
        paragraph_tokens = []
        offset = 0
        for sentence in _split_paragraph(text):
            sentence_tokens = tokenizer.tokenize(sentence)
            paragraph_tokens.extend(sentence_tokens)
            sequence_spans = slide_over_sequence(sentence_tokens, stop_signals=stop_signals,
                    window_size=window_size, seq_overlap=seq_overlap)
            paragraph_spans.extend([ [span[0]+offset, span[1]+offset, span[2], span[3]] 
                    for span in sequence_spans])
            offset += len(sentence_tokens)
        return paragraph_spans, paragraph_tokens
    else:
        tokens = tokenizer.tokenize(text)
        return [[0, len(tokens), 0, len(tokens)]], tokens
    
def _split_paragraph(para):
    """
    中文分句，【。！？省略号】，分句结果应该延后到双引号结束后。
    # --------------------- 
    # 作者：blmoistawinde 
    # 来源：CSDN 
    # 原文：https://blog.csdn.net/blmoistawinde/article/details/82379256 
    # 版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    # para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略。
    return para.split("\n")


def slide_over_sequence(sequence, window_size, seq_overlap, stop_signals):
    """
    There are sequence that are longer than the maximum sequence length.
    To deal with this, a sliding window approach is applied to take chunks
    of the up to max length with a overlap of `seq_overlap`.
    return sequence_spansthe contains actual the (left, right, start, end) indice.
    return:
        sequence_spans: List[left: int, right: int, start: int, end: int]
            left, right: the position span in the original sequence 
            start, end: the actual sequence span we care about (ingore overlap)


    window_size = 10,
    seq_overlap = 5,
    stop_signals = [".", ","]
    sequence:      A b c d e f g h i . J  k  l  m  n  o  p  ,  q  r  s  t  .
    idx:           0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
    0, 10, 0, 10   A b c d e f g h i .
                   |                   |
    5, 15, 5, 10             f g h i . J  k  l  m  n
                                       |              |
    10, 18, 5, 8                       J  k  l  m  n  o  p  , 
                                                      |        |
    13, 23, 5, 10                               m  n  o  p  ,  q  r  s  t  .
                                                               |             |
    return:
        sequence_spans = [[0, 10, 0, 10], [5, 15, 5, 9], 
                        [14, 23, 5, 9], [13, 23, 5, 10]]
    """

    def _backward(idx, stop_signal=None):
        if idx >= len(sequence):
            return len(sequence)
        while(stop_signal and idx > 0 and sequence[idx] != stop_signal):
            idx -= 1
        return idx

    stop_signals.append(None)
    cur_signal = 0
    sequence_spans = []
    left = 0
    right = 0
    while right < len(sequence):
        # each time find a span        
        _right = _backward(left + window_size, stop_signals[cur_signal])
        if _right <= right: # dead loop, degenerate stop_signal
            cur_signal += 1
            continue
            
        start = right - left
        end   = _right - left
        right = _right
        sequence_spans.append([left, right, start, end])      
        cur_signal = 0
        left = _backward(right - min(seq_overlap, (right-left)//2))        
    return sequence_spans


def reconstruct_entity(input_examples, entitys_iter):
    """ the entitys_iter contains the prediction entity of the splited examples.
    We need to reconstruct the complete entitys for each example in input_examples.
    and return the results as dictionary.
    input_examples: each should contains (start, end) indice.
    entitys_iter: iterator of entitys

    Overlaps follows first in first set order:
    --------------------------------------
    O   O O B-PER I-PER
          O O     O     O B-GPE I-GPE     
                        O B-LOC I-LOC O O
    --------------------------------------
    O   O O B-PER I-PER O B-GPE I-GPE O O
    --------------------------------------

    return: the complete entitys of each input example.
    """
    predict_entitys = []
    for i, example in enumerate(input_examples):
      _entity = []
      for span in example.sentence_spans:
        _, _, start, end = span
        # +1 to skip the first padding
        _entity.extend(next(entitys_iter)[start : end])     
      predict_entitys.append(_entity)
    assert len(predict_entitys) == len(input_examples)
    return predict_entitys


def reconstruct_from_estimator(input_examples, predict_results):
    """ the predict_results contains the prediction entity of the splited examples.
    We need to reconstruct the complete entitys for each example in input_examples.
    and return the results as dictionary.
    input_examples: each should contains (start, end) indice.
    predict_results: predict output of tensorflow estimator
                     estimator.predict(input_fn, yield_single_examples=True)

    Overlaps follows first in first set order:
    --------------------------------------
    O   O O B-PER I-PER
          O O     O     O B-GPE I-GPE 
                        O B-LOC I-LOC O O
    --------------------------------------
    O   O O B-PER I-PER O B-GPE I-GPE O O
    --------------------------------------

    return: the complete entitys of each input example.
    """
    predict_entitys = []
    for i, example in enumerate(input_examples):
      _entity = []
      for span in example.sentence_spans:
        _, _, start, end = span
        _entity.extend(next(predict_results)["predictions"][start : end])
      assert len(_entity) == len(example.text_entity)
      predict_entitys.append(_entity)
    assert len(predict_entitys) == len(input_examples)
    return predict_entitys


def bio_tags_to_spans(tag_sequence: List[str]):
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, 
    representing a single word span.
    Parameters
    ----------
    tag_sequence : List[str], required.
        The integer class labels for a sequence.
    Returns
    -------
    spans : List[TypedStringSpan]
        The typed, extracted spans from the sequence, 
        in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    left, right = 0, 1

    def _locate_next_idx(idx, mark, different=False):
        """ 
        if different is True:
            return the first position on the right of idx that is different to the mark.
        else versa
        """
        while (idx < len(tag_sequence) and tag_sequence[idx].startswith(mark) is different):
            idx += 1
        return idx

    def _add(left, right):
        _tag = list(set([tag[2:] for tag in tag_sequence[left: right]]))
        assert len(_tag) == 1
        spans.add((_tag[0], (left, right)))

    while left < len(tag_sequence):
        if tag_sequence[left].startswith('B'):
            if right < len(tag_sequence) and tag_sequence[right].startswith('I'):
                right = _locate_next_idx(right, 'I', different=True) # get next 'B-" or 'O"
            _add(left, right)
            left = right
            right += 1
        else:
            left = _locate_next_idx(left, 'B', different=False) # get next 'B-'
            right = left + 1

    return spans


    
TypedStringSpan = Tuple[str, Tuple[int, int]]

class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return ' '.join(self.tag_sequence)

class SpanBasedF1Measure:
    """
    Modified from https://github.com/ericput/bert-ner
    
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model performance during training. This metric
    works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O" if using the "BIO" label encoding).
    """
    def __init__(self):
        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self.all_tags: Set[str] = set()

    def __call__(self,
                 predictions: List[List[str]],
                 gold_labels: List[List[str]]):
        assert len(predictions) == len(gold_labels)
        # Iterate over timesteps in batch.
        for i in range(len(gold_labels)):
            predicted_spans = self._bio_tags_to_spans(predictions[i])
            if len(predictions[i]) != len(gold_labels[i]):
                print('The length of line %d are not equal: %d %d' % 
                    (i, len(predictions[i]), len(gold_labels[i])))

            gold_spans = self._bio_tags_to_spans(gold_labels[i])

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1 : float
        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1 for all spans.
        """
        self.all_tags.update(self._true_positives.keys())
        self.all_tags.update(self._false_positives.keys())
        self.all_tags.update(self._false_negatives.keys())
        all_metrics = defaultdict(lambda: {"f1": 0, "precision": 0, "recall": 0, })  
        for tag in self.all_tags:
            precision, recall, f1, support = self._compute_metrics(
                self._true_positives[tag],
                self._false_positives[tag],
                self._false_negatives[tag])
            all_metrics[tag]["precision"] = precision
            all_metrics[tag]["recall"] = recall
            all_metrics[tag]["f1"] = f1
            all_metrics[tag]["support"] = support

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1, support = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()))
        all_metrics["ALL"]["precision"] = precision
        all_metrics["ALL"]["recall"] = recall
        all_metrics["ALL"]["f1"] = f1
        all_metrics["ALL"]["support"] = support
        return all_metrics

    def log_measure(self, output_file, all_metrics=None):
        if not all_metrics:
            all_metrics = self.get_metric()
        all_tags = ['ALL'] + [tag for tag in sorted(self.all_tags)]
        with open(output_file, 'a', encoding='utf-8') as writer:
            writer.write("Eval results {}\n".format(datetime.now()))
            df = pd.DataFrame({
            'F1': [all_metrics[tag]["f1"] for tag in all_tags],
            'P': [all_metrics[tag]["precision"] for tag in all_tags],
            'R': [all_metrics[tag]["recall"] for tag in all_tags],
            'Support': [all_metrics[tag]["support"] for tag in all_tags],
            })
            df.set_index([all_tags], inplace=True)
            df.to_string(writer)
            writer.write("\n")
            return df
        
    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        stable_num = 1e-13
        precision = float(true_positives) / float(true_positives 
            + false_positives + stable_num)
        recall = float(true_positives) / float(true_positives 
            + false_negatives + stable_num)
        f1 = 2. * ((precision * recall) / (precision + recall + stable_num))
        support = true_positives + false_negatives
        return precision, recall, f1, support

    @staticmethod
    def _bio_tags_to_spans(tag_sequence: List[str]):
        """
        Given a sequence corresponding to BIO tags, extracts spans.
        Spans are inclusive and can be of zero length, 
        representing a single word span.
        Parameters
        ----------
        tag_sequence : List[str], required.
            The integer class labels for a sequence.
        Returns
        -------
        spans : List[TypedStringSpan]
            The typed, extracted spans from the sequence, 
            in the format (label, (span_start, span_end)).
            Note that the label `does not` contain any BIO tag prefixes.
        """
        spans: Set[Tuple[str, Tuple[int, int]]] = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, string_tag in enumerate(tag_sequence):
            # Actual BIO tag.
            bio_tag = string_tag[0]
            if bio_tag not in ["B", "I", "O"]:
                raise InvalidTagSequence(tag_sequence)
            conll_tag = string_tag[2:]
            if bio_tag == "O":
                # The span has ended.
                if active_conll_tag is not None:
                    if conll_tag == active_conll_tag:
                        spans.add((active_conll_tag, (span_start, span_end+1)))
                    else:
                        spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "B":
                # We are entering a new span; reset indices
                # and active tag to new span.
                if active_conll_tag is not None:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                # This is the case the bio label is an "I", but either:
                # 1) the span hasn't started - i.e. an ill formed span.
                # 2) The span is an I tag for a different conll annotation.
                # We'll process the previous span if it exists, but also
                # include this span. This is important, because otherwise,
                # a model may get a perfect F1 score whilst still including
                # false positive ill-formed spans.
                if active_conll_tag is not None:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
        # Last token might have been a part of a valid span.
        if active_conll_tag is not None:
            spans.add((active_conll_tag, (span_start, span_end)))
        return list(spans)


class UnbalancedMetrics(object):
    ''' Metric class to support multi-label classification evaluation.
        confusion matrix, 
        precision, recall and fbeta score 
    '''
    @staticmethod
    def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
        """Calculate a streaming confusion matrix.
        Calculates a confusion matrix. For estimation over a stream of data,
        the function creates an  `update_op` operation.
        Args:
            labels: A `Tensor` of ground truth labels with shape [batch size] and of
            type `int32` or `int64`. The tensor will be flattened if its rank > 1.
            predictions: A `Tensor` of prediction results for semantic labels, whose
            shape is [batch size] and type `int32` or `int64`. The tensor will be
            flattened if its rank > 1.
            num_classes: The possible number of labels the prediction task can
            have. This value must be provided, since a confusion matrix of
            dimension = [num_classes, num_classes] will be allocated.
            weights: Optional `Tensor` whose rank is either 0, or the same rank as
            `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `labels` dimension).
        Returns:
            total_cm: A `Tensor` representing the confusion matrix.
            update_op: An operation that increments the confusion matrix.
        """
        from tensorflow.python.ops import math_ops, metrics_impl
        total_cm, update_op = metrics_impl._streaming_confusion_matrix(
            labels=labels, predictions=predictions, num_classes=num_classes, 
            weights=weights)
        return math_ops.to_int32(total_cm), update_op

    @staticmethod
    def eval(total_cm, labels, beta=1):
        num_class = len(labels)
        precisions = []
        recalls = []
        fbetas = []
        supports = []
        beta = beta * beta
        for i, label in enumerate(labels):
            rowsum, colsum = np.sum(total_cm[i]), np.sum(total_cm[r][i] for r in range(num_class))
            precision = total_cm[i][i] / float(colsum+1e-12)
            recall = total_cm[i][i] / float(rowsum+1e-12)
            f = (1 + beta) * precision * recall / (beta * precision + recall+1e-12)
            precisions.append(precision)
            recalls.append(recall)
            fbetas.append(f)
            supports.append(int(rowsum))
        return precisions, recalls, fbetas, supports

def write_eval(writer, result, label_list=[], pos_labels=[]):
    """
    Write the evaluation results to the writer
        writer: the writer object
        result: the evaluation result as dictionary
        label_list: all possible labels
        pos_labels: postive labels that contributes to the weighted-average metrics
    """
    
    writer.write("Eval results {}\n".format(datetime.now()))
    for key in sorted(result.keys()):
        if key != "confusion_matrix":
            writer.write("%s = %s\n" % (key, str(result[key])))
    if "confusion_matrix" in result:
        if len(pos_labels) == 0 and len(label_list) > 0:
            pos_labels = label_list
        pos_idx = [i for i,x in enumerate(label_list) if x in pos_labels]
        confusion_matrix = result["confusion_matrix"]
        precisions, recalls, fbetas, supports = UnbalancedMetrics.eval(
                confusion_matrix, label_list)
        precisions, recalls = np.array(precisions), np.array(recalls)
        fbetas = np.array(fbetas)
        supports = np.array(supports)
        
        neg_labels = [label for label in label_list if label not in pos_labels]
        neg_idx = [i for i,x in enumerate(label_list) if x not in pos_labels]
        df = pd.DataFrame({
            'F1': fbetas[neg_idx],
            'P': precisions[neg_idx], 
            'R': recalls[neg_idx],
            'Support': supports[neg_idx],
            })
        df.set_index([neg_labels], inplace=True)
        df.Support = df.Support.astype(np.int32)
        df.to_string(writer)
        writer.write("\n")
        
        weights = supports[pos_idx]
        df = pd.DataFrame({
            'F1': np.append(np.average(fbetas[pos_idx], weights=weights), fbetas[pos_idx]),
            'P': np.append(np.average(precisions[pos_idx], weights=weights), precisions[pos_idx]), 
            'R': np.append(np.average(recalls[pos_idx], weights=weights), recalls[pos_idx]),
            'Support': np.append(np.sum(supports[pos_idx]), supports[pos_idx]),
            })
        df.set_index([['All'] + pos_labels], inplace=True)
        df.Support = df.Support.astype(np.int32)
        df.to_string(writer)
        writer.write("\n")
        return df
    else: return '\n' 

def write_confusion_matrix(path, confusion_matrix, label_list):
    " write confusion matrix as figure. "
    import matplotlib.pyplot as plt
    import seaborn as sns
    # df_cm = pd.DataFrame(confusion_matrix, dtype=np.int8, 
    #         index=label_list, columns=label_list)
    # sns.heatmap(df_cm, annot=True)
    plt.figure(figsize = (25, 10))
    sns.heatmap(confusion_matrix, xticklabels=label_list, yticklabels=label_list, 
            annot=True, fmt="d", cmap="YlGnBu")
    # ax.set_yticklabels(label_list, rotation=0)
    plt.title("confusion matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(path, #dpi=1000, 
        bbox_inches='tight')


def is_alpha(string):
    "check if string is composed of English characters only"
    try:
        return string.encode('ascii').isalpha()
    except:
        return false


def is_valid_bio(entity):
    """ check if the entity sequence is valid, 
    'I-' should not at the beginning, and should never follow 'O' neither. 
    """
    for j, e in enumerate(entity):
        if e.startswith('I-') and ( j == 0 or e[2:] != entity[j - 1][2:]):
            print(j, entity[j-1], e,)
            return False
    return True


def is_duplicate(data):
    "check if there are duplicate elements in data. "
    cnt = set(data)
    print("There are {} duplicate elements".format(int(len(data)-len(cnt))))
    return len(cnt) != len(data)


def to_json_string(_dict):
    """Serializes this instance to a JSON string."""
    return json.dumps(copy.deepcopy(_dict), indent=2, sort_keys=True) + "\n"


def cnt_entity(data):
    "helper func to count the entity in dataset"
    sb_measure = SpanBasedF1Measure()
    label = [example['entity'] for example in data]
    pred = label
    sb_measure(pred, label)
    sb_measure_file = os.path.join("./sb_measure.txt")
    print(sb_measure.log_measure(sb_measure_file))	

class Tests(unittest.TestCase):

    def test(self,):
        test = read_json_list('./data/test_ner.json')
        tic = time.time()
        cnt = 0
        for example in test:
            text = example['text']
            cnt += 1
            spans, paragraph_tokens = split_paragraph(text, tokenizer, seq_overlap=24)
        print("Efficiency: {}s/text".format((time.time()-tic)/cnt))


if __name__ == '__main__':
    import tokenization 
    from timeit import default_timer
    vocab_file = './chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.ChineseTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    # unittest.main()
    test = read_text('./data/test.txt')
    tic = default_timer()
    for _ in range(1000):
        for example in test:
            spans, text_tokens = split_paragraph(example, tokenizer, seq_overlap=24)
            # spans2, text_tokens = split_paragraph_2(example, tokenizer, seq_overlap=24)
    toc = default_timer()
    print("split_paragraph ")
    print(toc-tic)
        # for span in spans:
        #     left, right, _, _ = span
        #     tokens = text_tokens[left: right]
        #     print(tokens) 

