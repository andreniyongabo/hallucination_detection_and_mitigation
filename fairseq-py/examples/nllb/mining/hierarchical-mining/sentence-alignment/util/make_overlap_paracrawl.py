#!/usr/bin/env python3

import argparse

from dp_utils import yield_overlaps


def read_in_multi_file(f_in_name):
    langs = list()
    docs = list()  # each doc is a list of lines
    do_not_merge = list() # lines that should not be merged with prior
    lines = list()
    urls = list()
    for line in open(f_in_name, 'rt', encoding="utf-8"):
        if line.startswith('Rzc2NodXR6IHVuZC9vZGVyIHNvbnN0aW'):
            _, lang_code, url = line.strip().split('\t')
            langs.append(lang_code)
            urls.append(url)
            docs.append( list() )
            do_not_merge.append( set() )
        elif line.startswith('___DO_NOT_MERGE___'):
            do_not_merge[-1].add(len(docs[-1]))
        else:
            docs[-1].append(line.strip())

    src_lang = langs[0]
    tgt_lang = langs[1]

    src_docs = list()
    tgt_docs = list()

    src_urls = list()
    tgt_urls = list()

    src_do_not_merge = list()
    tgt_do_not_merge = list()

    for ii, (lang, doc, url, merge) in enumerate(zip(langs, docs, urls, do_not_merge)):
        if ii % 2 == 0:
            assert (lang == src_lang)
            src_docs.append(doc)
            src_urls.append(url)
            src_do_not_merge.append(merge)
        else:
            assert (lang == tgt_lang)
            tgt_docs.append(doc)
            tgt_urls.append(url)
            tgt_do_not_merge.append(merge)

    return src_lang, tgt_lang, src_docs, tgt_docs, src_urls, tgt_urls, src_do_not_merge, tgt_do_not_merge


def go(out_base_name, f_in_name, num_overlaps):
    src_lang, tgt_lang, src_docs, tgt_docs, _, _, _, _ = read_in_multi_file(f_in_name)

    for lang, docs in [(src_lang, src_docs), (tgt_lang, tgt_docs)]:
        unique_lines = set()
        for doc_lines in docs:
            for out_line in yield_overlaps(lines=doc_lines, num_overlaps=num_overlaps):
                unique_lines.add(out_line)

        with open(out_base_name + '.' + lang, 'wt', encoding="utf-8") as fout:
            for line in unique_lines:
                fout.write(line + '\n')


def _main():
    parser = argparse.ArgumentParser('VecAlign: Sentence alignment using sentence embeddings and FastDTW',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_base_name', type=str,
                        help='output text file')

    parser.add_argument('num_overlaps', type=int,
                        help='output text file')

    parser.add_argument('input', type=str,
                        help='text files to make embeddings for')

    args = parser.parse_args()
    go(out_base_name=args.output_base_name, f_in_name=args.input, num_overlaps=args.num_overlaps)


if __name__ == '__main__':
    _main()
