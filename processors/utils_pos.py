# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import collections
import io
import sys
import copy
from .utils import load_char_to_ids_dict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, char_input_ids, start_ids, end_ids, input_ids, input_mask, segment_ids, label_ids):
        self.char_input_ids = char_input_ids
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                 words=words,
                                                 labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                #splits = line.split(" ")
                splits = line.split()
                word = splits[0].strip(" ")
                if len(word) < 1:
                    continue
                words.append(word)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 char_vocab_file="./data/dict/bert_char_vocab",
                                 model_type='bert'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    char2ids_dict = load_char_to_ids_dict(char_vocab_file=char_vocab_file)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        #print(f"{len(example.words)} words in example: {' '.join(example.words)}")
        #print(f"{len(example.labels)} labels in example: {' '.join(example.labels)}")
        for word, label in zip(example.words, example.labels):
            word = word.strip()
            if len(word) < 1:
                continue
            #print(f"word: {word} label: {label} len_word: {len(word)}")
            word_tokens = []
            if model_type == 'roberta':
                word_tokens = tokenizer.tokenize(word, add_prefix_space=True)
            else:
                word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            #print(f"{len(tokens)} tokens: {' '.join(tokens)}")
            #print(f"{len(label_ids)} labels: {' '.join(map(str, label_ids))}")

        #print(f"{len(tokens)} tokens after tokenize: {' '.join(tokens)}")
        #print(f"{len(label_ids)} labels after tokenize: {' '.join(map(str, label_ids))}")
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        #print(f"{len(tokens)} tokens after padding special words: {' '.join(tokens)}")
        #print(f"{len(label_ids)} labels after padding special words: {' '.join(map(str, label_ids))}")

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
        #print(f"{len(tokens)} tokens after padding CLS: {' '.join(tokens)}")
        #print(f"{len(label_ids)} labels after padding CLS: {' '.join(map(str, label_ids))}")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        assert len(input_ids) == len(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
        #print(f"{len(tokens)} tokens after padding: {' '.join(tokens)}")
        #print(f"{len(label_ids)} labels after padding: {' '.join(map(str, label_ids))}")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        #convert input_ids to char_input_ids
        #all_seq_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        char_ids = []
        start_ids = []
        end_ids = []
        char_maxlen = max_seq_length * 6
        #print(f"Input tokens: {' '.join(tokens)}")
        for idx, token in enumerate(tokens): #char_ids for debug
            if len(char_ids) >= char_maxlen:
                break
            token = token.strip("##")
            if token in [tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token,\
                tokenizer.cls_token, tokenizer.mask_token]:
                start_ids.append(len(char_ids))
                end_ids.append(len(char_ids))
                char_ids.append(0)
            else:
                for char_idx, c in enumerate(token):
                    if len(char_ids) >= char_maxlen:
                        break
                    
                    if char_idx == 0:
                        start_ids.append(len(char_ids))
                    if char_idx == len(token) - 1:
                        end_ids.append(len(char_ids))

                    if c in char2ids_dict:
                        cid = char2ids_dict[c]
                    else:
                        cid = char2ids_dict["<unk>"]
                    char_ids.append(cid)

            if len(char_ids) < char_maxlen:
                char_ids.append(0)

        if len(char_ids) > char_maxlen:
            char_ids = char_ids[:char_maxlen]
        else:
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
        while len(start_ids) < max_seq_length:
            start_ids.append(char_maxlen-1)
        while len(end_ids) < max_seq_length:
            end_ids.append(char_maxlen-1)

        assert len(char_ids) == char_maxlen
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length
        if True:
            print(f"char_ids: {' '.join(map(str, char_ids))}")
            print(f"start_ids: {' '.join(map(str, start_ids))}")
            print(f"end_ids: {' '.join(map(str, end_ids))}")

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", ' '.join([str(x) for x in tokens]))
            logger.info("char_ids: %s", " ".join([str(x) for x in char_ids[:100]]))
            logger.info("input_ids: %s", ' '.join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", ' '.join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", ' '.join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", ' '.join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(char_input_ids=char_ids,
                              start_ids = start_ids,
                              end_ids = end_ids,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
