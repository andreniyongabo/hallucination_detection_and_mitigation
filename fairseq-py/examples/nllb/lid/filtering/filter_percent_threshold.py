#!python3 -u


import argparse
import os
import re
import sys


args = None
CRUBADAN_PATH="/large_experiments/mmt/data/monolingual/lid/from-christophe/crubadan_clean/"


def filter():
    most_freq_file=CRUBADAN_PATH+f"{args.lang}-words.txt"

    if not os.path.exists(most_freq_file):
        print(f"{most_freq_file} does not exist", file=sys.stderr)
        exit(1)

    most_freq_words = set([l.rstrip() for l in open(most_freq_file, 'r').readlines()])

    for line in sys.stdin:
        line = line.rstrip()
        tokens = set(line.split())
        inters = tokens.intersection(most_freq_words)
        num = len(inters)
        den = len(tokens)
        if den == 0:
            continue
        percent = num/den
        size_percent = len(''.join(inters))/len(line)

        if percent > args.threshold and size_percent > args.size_threshold:
            if args.show_percentage:
                print(f"{percent:.5f} {num} {den}")
                print(f"{size_percent:.5f} {len(''.join(inters))} {len(line)}")
            print(line)



def main():
    global args

    parser = argparse.ArgumentParser(
        description="Filter language file based on the percentage of intersection with the most frequent tokens in that language"
    )
    parser.add_argument("--lang", type=str, help="language name iso-639-3")
    parser.add_argument("--threshold", type=float, help="percentage threshold. The line in stdin is kept if it contains more than threshold tokens that are in the most frequent tokens")
    parser.add_argument("--size-threshold", type=float, help="percentage size threshold. percentage = length of tokens intersection with the most frequent tokens / len(line)")
    parser.add_argument("--show-percentage", action="store_true")
    args = parser.parse_args()

    filter()


if __name__ == '__main__':
    main()