#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import re
import sys

args = None

# adapted from /private/home/edunov/t2t_filter/clean_histogram.py


def filter():
    def read_hist(f):
        ch=[]

        if args.histogram_threshold:
            ctr = 0
            counts = 0
            for line in f:
                ll = line.split()
                if len(ll) == 1:
                    counts += float(ll[0])
                else:
                    counts += float(ll[1])

            counts_threshold = counts * args.histogram_threshold
            counts = 0
            f.seek(0, 0)
            for line in f:
                c = line[0]
                ll = line.split()
                if len(ll) == 1:
                    counts += float(ll[0])
                else:
                    counts += float(ll[1])
                if counts > counts_threshold:
                    break
                ch.append(c)
        else:
            for line in f:
                c = line[0]
                if c == args.threshold_character:
                    break
                ch.append(c)
        return ch


    with(open("{}/{}".format(args.histograms, args.lang), 'r', encoding='utf8')) as f:
        ch = read_hist(f)

    print(f"Accepted characters: {ch}", file=sys.stderr)

    # ignore fastText style labels in the document
    prog = re.compile(r"__label__\w+ ")

    for line in sys.stdin:
        ln = line.strip()
        analyzed_ln = ln
        m = prog.match(analyzed_ln)
        if m:
            analyzed_ln = prog.sub(r'\0', analyzed_ln)

        cnt = len([c for c in analyzed_ln if c in ch])
        score = cnt / len(analyzed_ln)

        if args.show_score:
            print(f"{score:.6f}\t{ln}")
        else:
            if score >= args.threshold:
                print(ln)
            else:
                print(f"Rejected {score:.6f} {ln}", file=sys.stderr)


def main():
    global args

    parser = argparse.ArgumentParser(
        description="Filter language file based on the percentage of intersection with the most frequent tokens in that language"
    )
    parser.add_argument("--lang", type=str, help="language name 2 letters code")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold")
    parser.add_argument('--threshold-character', type=str, default=']', help='Threshold character')
    parser.add_argument("--histogram-threshold", type=float, default=0.0, help="threshold")
    parser.add_argument('--histograms', type=str, default='/checkpoint/edunov/cc60_multilingual/clean_hists/', help='Path to histograms')
    parser.add_argument("--show-score", action="store_true", help="display the accepted characters percentage for each line")

    args = parser.parse_args()

    filter()


if __name__ == '__main__':
    main()
