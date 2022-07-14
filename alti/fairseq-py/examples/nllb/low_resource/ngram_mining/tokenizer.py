import argparse
from nltk.tokenize import word_tokenize
import sys
import os
import glob


def process(args, paths):
    output_tok = os.path.join(args.output_dir, f"{args.shard_id}.tok")
    output_raw = os.path.join(args.output_dir, f"{args.shard_id}.raw")
    with open(output_tok, "w") as tok_out, open(output_raw, "w") as raw_out:
        i = 0
        processed = 0
        for path in paths:
            print("Processing", path)
            with open(path) as in_f:
                for line in in_f:
                    sentence = line.strip()
                    if i % args.num_shards == args.shard_id:
                        tokens = word_tokenize(sentence)
                        print(" ".join(tokens), file=tok_out)
                        print(sentence, file=raw_out)
                        processed += 1
                        if args.max_count is not None and processed >= args.max_count:
                            return processed
                    i += 1
    return processed


def process_src(args, paths):
    output_src = os.path.join(args.output_dir, f"{args.shard_id}.src")
    with open(output_src, "w") as src_out:
        i = 0
        processed = 0
        for path in paths:
            assert path.split(".")[-1] == args.tgt
            src_path = ".".join(path.split(".")[:-1]) + "." + args.src
            print("Processing src", src_path)
            with open(src_path) as in_f:
                for line in in_f:
                    sentence = line.strip()
                    if i % args.num_shards == args.shard_id:
                        print(sentence, file=src_out)
                        processed += 1
                        if args.max_count is not None and processed >= args.max_count:
                            return processed
                    i += 1
    return processed


def main(args):
    paths = sorted(glob.glob(args.paths))
    os.makedirs(args.output_dir, exist_ok=True)
    num_processed = process(args, paths)
    print(
        f"Done with {args.paths}. Processed {num_processed} lines (shard={args.shard_id}/{args.num_shards})"
    )

    if args.is_translated:
        num_src_processed = process_src(args, paths)
        print(
            f"Done with {args.paths}. Processed {num_processed} source lines (shard={args.shard_id}/{args.num_shards})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", default="/datasets01/ccmatrix.v1/texts/ccnet32.en0.txt"
    )
    parser.add_argument("--output-dir", default="ngram_mining_outputs/eng_txt")
    parser.add_argument("--num-shards", default=4, type=int)
    parser.add_argument("--shard-id", default=0, type=int)
    parser.add_argument("--is-translated", action="store_true")
    parser.add_argument("--src", default=None, type=str)
    parser.add_argument("--tgt", default=None, type=str)
    parser.add_argument("--max-count", default=None, type=int)

    main(parser.parse_args())
