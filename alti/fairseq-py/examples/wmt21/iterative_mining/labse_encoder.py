import argparse
from sentence_transformers import SentenceTransformer

import glob
import hashlib
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import numpy as np
import pickle

def generate_embeddings(args):
    embeddings = None
    model = SentenceTransformer('LaBSE')
    pool = model.start_multi_process_pool()
    sentences = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if args.remove_domain_tag:
                line = ' '.join(line.split(' ')[2:])
            sentences.append(line)
    embeddings = model.encode_multi_process(sentences, pool)
    with open(args.output_file, 'w') as out_f:
        embeddings.tofile(out_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default="/large_experiments/mmt/wmt21/monolingual_preprocessed/en/news.2020.en.shuffled.deduped.filtered/0.preprocessed")
    parser.add_argument('--output-file', default="/large_experiments/mmt/wmt21/labse_embeddings/monolingual/en/news.2020.en.shuffled.deduped.filtered/0.preprocessed")
    parser.add_argument('--batch-size', default=10000)
    parser.add_argument('--remove-domain-tag', default=True)
    args = parser.parse_args()
    generate_embeddings(args)
