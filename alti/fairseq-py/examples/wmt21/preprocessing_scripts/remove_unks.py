import argparse
from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary, data_utils, indexed_dataset


def main(args):
    tgt_dict = Dictionary.load(args.tgt_dict)
    src, tgt = args.direction.split('-')
    src_in = f"{args.input_prefix}.{src}"
    tgt_in = f"{args.input_prefix}.{tgt}"
    src_out = f"{args.output_prefix}.{src}"
    tgt_out = f"{args.output_prefix}.{tgt}"
    total = 0
    unks = 0

    with open(src_in) as src_in_f, open(tgt_in) as tgt_in_f:
        with open(src_out, 'w') as src_out_f, open(tgt_out, 'w') as tgt_out_f:
            for src_line, tgt_line in zip(src_in_f, tgt_in_f):
                if total % 200000  == 0:
                    print(f"{unks} unks line out of {total} lines")
                encoded_tgt = tgt_dict.encode_line(
                    line=tgt_line,
                    line_tokenizer=tokenize_line,
                    add_if_not_exist=False,
                    append_eos=True,
                    reverse_order=False)
                if tgt_dict.unk() not in encoded_tgt:
                    src_out_f.write(src_line)
                    tgt_out_f.write(tgt_line)
                else:
                    unks += 1
                total += 1
    print(f"Done filtering, {unks} unks line out of {total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-prefix',
            default='/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards/sharded_bin/shard000/train.sharded.en_zh')
    parser.add_argument('--output-prefix',
            default='/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards/sharded_bin/shard000/train.sharded.rm_unks.en_zh')
    parser.add_argument('--direction',
            default='en-zh')
    parser.add_argument('--tgt-dict',
            default='/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards/dict.zh.txt')
    args = parser.parse_args()
    main(args)
