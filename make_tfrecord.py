# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    make_tfrecord.py
   Description :
   Author :       Wings DH
   Time：         2020/1/18 9:49 上午
-------------------------------------------------
   Change Activity:
                   2020/1/18: Create
-------------------------------------------------
"""
import collections
import os

import tensorflow as tf
import data_processing
from data_processing import bert_tokenization

flags = tf.flags

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, masked_lm_ids, soft_mask_label, masked_lm_weights=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """

        self.guid = guid
        self.text = text
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.soft_mask_label = soft_mask_label

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [bert_tokenization.printable_text(x) for x in self.text]))
        s += "masked: %s\n" % (" ".join(
            [bert_tokenization.printable_text(x) for x in self.masked_lm_ids]))
        s += "softed: %s\n" % ("  ".join([str(x) for x in self.soft_mask_label]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def load_data(input_file, truth_file):
    """
    Load data from data files

    Parameters
    ----------
    input_file
    truth_file

    Returns
    -------

    """
    examples = []
    for i, (inp, trth) in enumerate(
            zip(open(input_file, encoding='utf-8'), open(truth_file, encoding='utf-8'))):
        inp_fields = inp.strip().split("\t")
        trth_fields = trth.strip().split(" ")

        guid = inp_fields[0]
        text = data_processing.bert_tokenization.convert_to_unicode(inp_fields[1])
        text = text[:FLAGS.max_seq_length - 2]
        masked_lm_tokens = [t for t in text]
        masked_lm_weights = [1.0 for _ in text]
        soft_mask_label = [0 for _ in text]
        for j, pos_tok in enumerate(trth_fields[1:]):
            if j % 2 == 0:
                pos = pos_tok.strip(",")
            else:
                # replace mispell tokens of the input sentence to correct tokens.
                if int(pos) < FLAGS.max_seq_length:
                    masked_lm_tokens[int(pos) - 1] = pos_tok.strip(",")
                    soft_mask_label[int(pos) - 1] = 1
        examples.append(
            InputExample(guid=guid,
                         text=text,
                         masked_lm_ids=masked_lm_tokens,
                         soft_mask_label=soft_mask_label,
                         masked_lm_weights=masked_lm_weights))
    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 masked_lm_ids,
                 masked_lm_weights,
                 soft_mask_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.soft_mask_label = soft_mask_label


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    ori_tokens = tokenizer.tokenize(example.text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    def create_tokens_and_segments():
        tokens = []
        _segment_ids = []
        tokens.append("[CLS]")
        _segment_ids.append(0)
        for token in ori_tokens:
            tokens.append(token)
            _segment_ids.append(0)
        tokens.append("[SEP]")
        _segment_ids.append(0)

        for j, tok in enumerate(tokens):
            if tok not in tokenizer.vocab:
                tokens[j] = "[UNK]"

        _input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        _input_mask = [1] * len(_input_ids)

        # Zero-pad up to the sequence length.
        while len(_input_ids) < max_seq_length:
            _input_ids.append(0)
            _input_mask.append(0)
            _segment_ids.append(0)
        return _input_ids, _segment_ids, _input_mask

    input_ids, segment_ids, input_mask = create_tokens_and_segments()

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    def create_masked_ids_and_weights():
        _masked_lm_ids = []
        _soft_mask_label = []
        _masked_lm_ids.append("[CLS]")
        for tok in example.masked_lm_ids:
            _masked_lm_ids.append(tok if tok in tokenizer.vocab else "[UNK]")
        _masked_lm_ids.append("[SEP]")
        _masked_lm_ids = tokenizer.convert_tokens_to_ids(_masked_lm_ids)

        _soft_mask_label.append(0)
        for label in example.soft_mask_label:
            _soft_mask_label.append(label)
        _soft_mask_label.append(0)

        _masked_lm_weights = [1.0] * len(_masked_lm_ids)
        _masked_lm_weights[0] = 0.0
        _masked_lm_weights[len(_masked_lm_ids) - 1] = 0.0
        while len(_masked_lm_ids) < max_seq_length:
            _masked_lm_ids.append(0)
            _masked_lm_weights.append(0.0)
            _soft_mask_label.append(0)
        return _masked_lm_ids, _masked_lm_weights, _soft_mask_label

    masked_lm_ids, masked_lm_weights, soft_mask_label = create_masked_ids_and_weights()

    assert len(masked_lm_ids) == max_seq_length
    assert len(masked_lm_weights) == max_seq_length
    assert len(soft_mask_label) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([bert_tokenization.printable_text(x) for x in example.text]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("masked_lm_ids: %s" % " ".join([str(x) for x in masked_lm_ids]))
        tf.logging.info("masked_lm_weights: %s" % " ".join([str(x) for x in masked_lm_weights]))
        tf.logging.info("soft_mask_label: %s" % " ".join([str(x) for x in soft_mask_label]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights,
        soft_mask_label=soft_mask_label)
    return feature


def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["masked_lm_ids"] = create_int_feature(feature.masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(feature.masked_lm_weights)
        features["soft_mask_label"] = create_float_feature(feature.soft_mask_label)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    tokenizer = bert_tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

    examples = load_data(FLAGS.input_file, FLAGS.truth_file)
    for example in examples[:10]:
        print(example)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    train_file = os.path.join(FLAGS.output_dir, "train_{0}.tf_record".format(FLAGS.max_seq_length))
    file_based_convert_examples_to_features(examples, FLAGS.max_seq_length, tokenizer, train_file)


def init_parameters():
    # Required parameters
    flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
    flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")

    # Optional parameters
    flags.DEFINE_string("task_name", None, "The name of the task to train.")
    flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
    flags.DEFINE_string("input_file", './data/TrainingInputAll.txt', "The input file path.")
    flags.DEFINE_string("truth_file", './data/TrainingTruthAll.txt', "The truth file path.")


if __name__ == "__main__":
    init_parameters()

    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")

    tf.app.run()
