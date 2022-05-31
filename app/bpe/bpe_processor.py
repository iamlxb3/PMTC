import os
import ntpath
import tqdm
import collections
from typing import List
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from .utils import load_save_json

_TOP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
__CN_VOCAB_PATH = os.path.join(_TOP_DIR, 'static', 'cn_vocab')


def _read_cn_chars():
    cn_vocabs = []
    with open(__CN_VOCAB_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cn_vocabs.append(line)
    cn_vocabs = tuple(cn_vocabs)
    return cn_vocabs


def create_func1(sep_token_id, cls_token_id):
    def get_special_tokens_mask(token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [sep_token_id, cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    return get_special_tokens_mask


def _convert_token_to_id_with_added_voc(token, added_tokens_encoder):
    if token is None:
        return None

    if token in added_tokens_encoder:
        return added_tokens_encoder[token]


def create_func2(added_tokens_encoder, mask_token_id):
    def convert_tokens_to_ids(tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens == '[MASK]':
            return mask_token_id

        if tokens is None:
            return None

        if isinstance(tokens, str):
            return _convert_token_to_id_with_added_voc(tokens, added_tokens_encoder)

        ids = []
        for token in tokens:
            ids.append(_convert_token_to_id_with_added_voc(token, added_tokens_encoder))
        return ids

    return convert_tokens_to_ids


def _tokenizer_post_process(tokenizer):
    tokenizer.max_len = len(tokenizer.get_vocab())
    tokenizer.cls_token_id = 0
    tokenizer.pad_token_id = 1
    tokenizer.sep_token_id = 2
    tokenizer.mask_token_id = 3
    tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
    tokenizer.added_tokens_encoder = {}
    tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)
    tokenizer.mask_token = '[MASK]'
    tokenizer._pad_token = '[PAD]'
    return tokenizer


class BpeProcessor:
    def __init__(self,
                 log_id_cn_char_path=None):
        if log_id_cn_char_path is not None:
            self.log_id_cn_char_dict = load_save_json(log_id_cn_char_path, 'load')
        else:
            self.log_id_cn_char_dict = None
        self.cn_chars = _read_cn_chars()
        # self._special_chars = ('<s>', '<pad>', '</s>', '<unk>', '<mask>', '_')

    @property
    def vocab_len(self):
        return len(self.cn_chars)

    def create_log_id_cn_char_mapping(self,
                                      raw_data_path=None,
                                      split_char=None,
                                      total_log_ids=None,
                                      cn_char_dict_save_path=None):

        # read total_log_ids
        if raw_data_path is not None:
            total_log_ids = []
            with open(raw_data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if not split_char:
                            log_ids = line.split()
                            total_log_ids.extend(log_ids)

        log_id_cn_char_dict = {}
        special_chars = ('<unk>',)

        for i, token in enumerate(special_chars):
            log_id_cn_char_dict[token] = self.cn_chars[i]

        total_log_ids = list(collections.Counter(total_log_ids).items())
        print(f"Number of unique log ids: {len(total_log_ids)}")
        total_log_ids = sorted(total_log_ids, key=lambda x: x[1], reverse=True)[:self.vocab_len - len(special_chars)]

        for i, (log_id, _) in enumerate(total_log_ids):
            log_id_cn_char_dict[log_id] = self.cn_chars[i + len(special_chars)]

        if cn_char_dict_save_path is None:
            cn_char_dict_save_path = os.path.join(_CURRENT_DIR_PATH, 'static',
                                                  f'log_id_cn_char_{len(log_id_cn_char_dict)}.json')
        load_save_json(cn_char_dict_save_path, 'save', data=log_id_cn_char_dict)
        self.log_id_cn_char_dict = log_id_cn_char_dict
        return log_id_cn_char_dict, cn_char_dict_save_path

    def convert_raw_data_to_cn_char(self,
                                    raw_id_sample: List[str]):
        unk_char = self.log_id_cn_char_dict['<unk>']
        assert self.log_id_cn_char_dict is not None
        return [self.log_id_cn_char_dict.get(x, unk_char) for x in raw_id_sample]

    def convert_raw_data_to_cn_char_data_from_arr(self,
                                                  seq_arr,
                                                  save_file_path=None):
        with open(save_file_path, 'w') as save_f:
            for seq_x in tqdm.tqdm(seq_arr, total=len(seq_arr)):
                seq_x = [x for x in seq_x if x != '[PAD]']
                line_cn_chars = self.convert_raw_data_to_cn_char(seq_x)
                save_f.write(''.join(line_cn_chars) + '\n\n')
        print(f"Save cn char file to {save_file_path}")
        return save_file_path

    def convert_raw_data_to_cn_char_data_from_file(self,
                                                   raw_data_path,
                                                   save_file_path=None,
                                                   split_char=None):

        if save_file_path is None:
            save_dir = os.path.dirname(raw_data_path)
            save_name = ntpath.basename(raw_data_path)
            save_file_path = os.path.join(save_dir, save_name.split('.')[0] + '_cnchar.' + save_name.split('.')[1])

        read_f_tqdm = tqdm.tqdm(total=os.path.getsize(raw_data_path))

        with open(save_file_path, 'w') as save_f:
            with open(raw_data_path, 'r') as read_f:
                for line in read_f:
                    line = line.strip()
                    if line:
                        if not split_char:
                            line_log_ids = line.split()
                        else:
                            raise NotImplementedError
                        line_cn_chars = self.convert_raw_data_to_cn_char(line_log_ids)
                    read_f_tqdm.update(len(line))
                    save_f.write(''.join(line_cn_chars) + '\n\n')
        print(f"Save cn char file to {save_file_path}")
        return save_file_path

    def train_tokenizer(self, cn_char_path, vocab_size, save_path, log_id_cn_char_dict):

        special_chars = ('<s>', '<pad>', '</s>', '<mask>', '_')
        tokenizer = Tokenizer(BPE())
        # tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=list(special_chars))
        tokenizer.train(files=[cn_char_path], trainer=trainer)

        vocab = tokenizer.get_vocab()
        # vocab_re = {v: k for k, v in vocab.items()}
        unk_cn_char = log_id_cn_char_dict['<unk>']
        if unk_cn_char not in vocab:
            tokenizer.add_tokens([unk_cn_char])
        # vocab = list(tokenizer.get_vocab().items())
        # vocab = sorted(vocab, key=lambda x: x[1])

        tokenizer.save(save_path, pretty=True)
        print(f"Save tokenizer to {save_path}, vocab size: {tokenizer.get_vocab_size()}")
        return _tokenizer_post_process(tokenizer)

    def load_tokenizer(self, path):
        tokenizer = Tokenizer.from_file(path)
        print(f"Load tokenizer from {path}, vocab size: {tokenizer.get_vocab_size()}")
        return _tokenizer_post_process(tokenizer)
