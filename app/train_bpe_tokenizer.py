import os

from bpe.bpe_processor import BpeProcessor

import argparse


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--raw_data_path', type=str, required=True)
    parser.add_argument('--log_id_map_path', type=str, required=True)
    parser.add_argument('--tokenizer_save_path', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=50000)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    vocab_size = args.vocab_size
    raw_data_path = args.raw_data_path
    log_id_map_path = args.log_id_map_path
    tokenizer_save_path = args.tokenizer_save_path
    # init processor
    bpe_processor = BpeProcessor()

    # raw data path

    # create cn_char mapping
    log_id_cn_char_dict, cn_char_dict_save_path = bpe_processor.create_log_id_cn_char_mapping(raw_data_path,
                                                                                              cn_char_dict_save_path=log_id_map_path)

    # convert raw data to cn char data
    cn_char_data_save_path = bpe_processor.convert_raw_data_to_cn_char_data_from_file(raw_data_path)

    # train & save tokenizer
    tokenizer = bpe_processor.train_tokenizer(cn_char_data_save_path,
                                              vocab_size,
                                              tokenizer_save_path,
                                              log_id_cn_char_dict)
    print(f"Save tokenizer path to {os.path.abspath(tokenizer_save_path)}")


if __name__ == '__main__':
    main()
