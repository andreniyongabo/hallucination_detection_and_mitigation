#!/usr/bin/env python3

import argparse
import pickle
from math import ceil
from random import seed as seed

import numpy as np

from dp_utils import make_alignment_types, print_alignments, print_all_scores, read_alignments, \
    read_in_embeddings, make_doc_embedding, vecalign
from logger import logger
from score import score_multiple, log_final_scores


from vecalign import add_debug_args

from make_overlap_paracrawl import read_in_multi_file


def _main():
    # make runs consistent
    seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser('VecAlign: Sentence alignment using sentence embeddings and FastDTW',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('multi_file', type=str, help='paracrawl format file')
    parser.add_argument('base_name', type=str, help='expects *.lc, *.lc.emb for lc in src and tgt lang')
    parser.add_argument('--print_all_scores', default=False, action='store_true',  help='report all matching scores of final pass')

    add_debug_args(parser)

    args = parser.parse_args()

    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    if args.alignment_max_size < 2:
        logger.warning('Alignment_max_size < 2. Increasing to 2 so that 1-1 alignments will be considered')
        args.alignment_max_size = 2

    src_lang, tgt_lang, src_docs, tgt_docs, src_urls, tgt_urls, src_do_not_merge, tgt_do_not_merge = read_in_multi_file(args.multi_file)

    src_embed_txt = args.base_name + '.' + src_lang
    src_embed = src_embed_txt + '.emb'
    tgt_embed_txt = args.base_name + '.' + tgt_lang
    tgt_embed = tgt_embed_txt + '.emb'

    src_sent2line, src_line_embeddings = read_in_embeddings(src_embed_txt, src_embed)
    tgt_sent2line, tgt_line_embeddings = read_in_embeddings(tgt_embed_txt, tgt_embed)

    width_over2 = ceil(args.alignment_max_size / 2.0) + args.search_buffer_size

    test_alignments = []
    stack_list = []
    for src_lines, tgt_lines, src_url, tgt_url, src_merge, tgt_merge in zip(src_docs, tgt_docs, src_urls, tgt_urls, src_do_not_merge, tgt_do_not_merge):
        logger.info('Aligning...')

        logger.info('Making source embeddings')
        vecs0 = make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, args.alignment_max_size)
        logger.info('Making target embeddings')
        vecs1 = make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.alignment_max_size)

        logger.warning('testing %s %s', vecs0.shape, vecs1.shape)

        final_alignment_types = make_alignment_types(args.alignment_max_size)
        logger.debug('Considering alignment types %s', final_alignment_types)

        stack = vecalign(vecs0=vecs0,
                         vecs1=vecs1,
                         do_not_merge0=src_merge, 
                         do_not_merge1=tgt_merge,
                         final_alignment_types=final_alignment_types,
                         del_percentile_frac=args.del_percentile_frac,
                         width_over2=width_over2,
                         max_size_full_dp=args.max_size_full_dp,
                         costs_sample_size=args.costs_sample_size,
                         num_samps_for_norm=args.num_samps_for_norm)

        # write final alignments to stdout
        print('Rzc2NodXR6IHVuZC9vZGVyIHNvbnN0aW %s %s'%(src_url, tgt_url))
        print_alignments(stack[0]['final_alignments'], stack[0]['alignment_scores'])
        if args.print_all_scores:
            print_all_scores(stack[0]['b_offset'],stack[0]['a_b_costs'],stack[0]['alignment_types'])

        test_alignments.append(stack[0]['final_alignments'])
        stack_list.append(stack)

    if args.debug_save_stack:
        pickle.dump(stack_list, open(args.debug_save_stack, 'wb'))


if __name__ == '__main__':
    _main()
