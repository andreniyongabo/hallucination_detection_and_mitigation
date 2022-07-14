import faiss
import numpy as np
import datetime
import glob
import argparse
import pickle
import os
from subprocess import check_call


GB = 1024*1024*1024


def call(cmd):
    print(cmd)
    check_call(cmd, shell=True)


def pprint(s, stdout=None):
    if stdout is not None:
        print(f"[{datetime.datetime.now()}] {s}", file=stdout)
        stdout.flush()
    print(f"[{datetime.datetime.now()}] {s}")


def get_batches(directory, lang, stdout=None, prefix='all_avg_pool'):
    pprint(f"Finding in {directory}/{prefix}.{lang}*", stdout)
    files = glob.glob(f'{directory}/{prefix}.{lang}*')
    emb_files = []
    txt_files = []
    for emb_fi in files:
        emb_files.append(emb_fi)
        txt_fi = emb_fi.replace(prefix, 'sentences')
        txt_files.append(txt_fi)
    return emb_files, txt_files


def load_batch(emb_file, dim):
    embeddings = np.fromfile(emb_file, dtype=np.float32)
    num_rows = int(embeddings.shape[0] / dim)
    embeddings = embeddings.reshape((num_rows, dim))
    faiss.normalize_L2(embeddings)
    return embeddings


def knnGPU_sharded(x_batches_f, y_batches_f, args, stdout=None, direction='x2y'):
    sims = []
    inds = []
    xfrom = 0
    xto = 0
    dim = args.dim
    k = args.neighborhood
    for i, x_batch_f in enumerate(x_batches_f):
        yfrom = 0
        yto = 0
        x_batch = load_batch(x_batch_f, dim)
        xto = xfrom + x_batch.shape[0]

        if args.num_shards is not None and args.shard_id is not None:
            if i % args.num_shards != args.shard_id:
                pprint(f"Skipping {xfrom}-{xto}", stdout=stdout)
                xfrom = xto
                continue
        if os.path.exists(args.output+f'/sim_batch_{direction}_{xfrom}_{xto}.bin') \
            and os.path.exists(args.output+f'/ind_batch_{direction}_{xfrom}_{xto}.bin'):
            sim_batch = pickle.load(open(args.output+f'/sim_batch_{direction}_{xfrom}_{xto}.bin', 'rb'))
            ind_batch = pickle.load(open(args.output+f'/ind_batch_{direction}_{xfrom}_{xto}.bin', 'rb'))
            pprint(f"Loaded {direction} {xfrom}-{xto}", stdout=stdout)
        else:
            bsims, binds= [], []
            for y_batch_f in y_batches_f:
                y_batch = load_batch(y_batch_f, dim)
                neighbor_size = min(k, y_batch.shape[0])
                yto = yfrom + y_batch.shape[0]
                pprint('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto), stdout=stdout)
                idx = faiss.IndexFlatIP(dim)
                idx = faiss.index_cpu_to_all_gpus(idx)
                idx.add(y_batch)
                bsim, bind = idx.search(x_batch, neighbor_size)

                bsims.append(bsim)
                binds.append(bind + yfrom)
                yfrom += y_batch.shape[0]
                del idx
                del y_batch
            bsims = np.concatenate(bsims, axis=1)
            binds = np.concatenate(binds, axis=1)
            aux = np.argsort(-bsims, axis=1)
            sim_batch = np.zeros((x_batch.shape[0], k), dtype=np.float32)
            ind_batch = np.zeros((x_batch.shape[0], k), dtype=np.int64)
            for i in range(x_batch.shape[0]):
                for j in range(k):
                    sim_batch[i, j] = bsims[i, aux[i, j]]
                    ind_batch[i, j] = binds[i, aux[i, j]]
            pickle.dump(sim_batch, open(args.output+f'/sim_batch_{direction}_{xfrom}_{xto}.bin', 'wb'))
            pickle.dump(ind_batch, open(args.output+f'/ind_batch_{direction}_{xfrom}_{xto}.bin', 'wb'))
        sims.append(sim_batch)
        inds.append(ind_batch)
        xfrom += x_batch.shape[0]
        del x_batch
    if len(sims) == 0 or len(inds) == 0:
        return None, None
    sim = np.concatenate(sims, axis=0)
    ind = np.concatenate(inds, axis=0)
    return sim, ind


def score(sim, fwd_mean, bwd_mean, margin):
    return margin(sim, (fwd_mean + bwd_mean) / 2)


def score_candidates(sim_mat, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False,
        stdout=None):
    pprint(' - scoring {:d} candidates'.format(sim_mat.shape[0]), stdout=stdout)
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = int(candidate_inds[i, j])
            scores[i, j] = score(sim_mat[i, j], fwd_mean[i], bwd_mean[k], margin)
    return scores

def load_text(files, stdout=None):
    all_sentences = []
    for fi in files:
        with open(fi) as sentence_fi:
            for line in sentence_fi:
                all_sentences.append(line.strip())
    pprint(f"Read {len(all_sentences)} sentences")
    return all_sentences


def main(args):
    stdout = None
    if args.stdout is not None:
        stdout = open(args.stdout, mode='w', encoding='utf-8', errors='surrogateescape')

    pprint(f'Start', stdout)
    x_batches_f, x_sents_f = get_batches(args.src_dir, args.src_lang, stdout)
    y_batches_f, y_sents_f = get_batches(args.tgt_dir, args.tgt_lang, stdout)
    margin = lambda a, b: a / b
    pprint(x_batches_f, stdout)
    pprint(x_sents_f, stdout)
    pprint(y_batches_f, stdout)
    pprint(y_sents_f, stdout)
    if os.path.exists(args.output+'/scores.bin') \
            and os.path.exists(args.output+'/indices.bin'):
        scores = pickle.load(open(args.output+'/scores.bin', 'rb'))
        indices = pickle.load(open(args.output+'/indices.bin', 'rb'))
    else:
        if os.path.exists(args.output+'/y2x_sim.bin') \
                and os.path.exists(args.output+'/y2x_ind.bin'):
            y2x_ind = pickle.load(open(args.output+'/y2x_ind.bin', 'rb'))
            y2x_sim = pickle.load(open(args.output+'/y2x_sim.bin', 'rb'))
            pprint("Loaded y2x_sim, y2x_ind", stdout)
        else:
            y2x_sim, y2x_ind = knnGPU_sharded(
                y_batches_f, x_batches_f,
                args,
                stdout=stdout,
                direction='y2x')
            if args.num_shards is None and args.shard_id is None:
                pickle.dump(y2x_sim, open(args.output+'/y2x_sim.bin', 'wb'))
                pickle.dump(y2x_ind, open(args.output+'/y2x_ind.bin', 'wb'))
        if os.path.exists(args.output+'/x2y_sim.bin') \
                and os.path.exists(args.output+'/x2y_ind.bin'):
            x2y_ind = pickle.load(open(args.output+'/x2y_ind.bin', 'rb'))
            x2y_sim = pickle.load(open(args.output+'/x2y_sim.bin', 'rb'))
            pprint("Loaded x2y_sim, x2y_ind", stdout)
        else:
            x2y_sim, x2y_ind = knnGPU_sharded(
                x_batches_f, y_batches_f,
                args,
                stdout=stdout,
                direction='x2y')
            if args.num_shards is None and args.shard_id is None:
                pickle.dump(x2y_sim, open(args.output+'/x2y_sim.bin', 'wb'))
                pickle.dump(x2y_ind, open(args.output+'/x2y_ind.bin', 'wb'))
        if args.num_shards is not None and args.shard_id is not None:
            return

        x2y_mean = x2y_sim.mean(axis=1)
        y2x_mean = y2x_sim.mean(axis=1)
        fwd_scores = score_candidates(x2y_sim, x2y_ind, x2y_mean, y2x_mean, margin,
                stdout=stdout)
        bwd_scores = score_candidates(y2x_sim, y2x_ind, y2x_mean, x2y_mean, margin,
                stdout=stdout)
        fwd_best = x2y_ind[np.arange(x2y_sim.shape[0]), fwd_scores.argmax(axis=1)]
        bwd_best = y2x_ind[np.arange(y2x_sim.shape[0]), bwd_scores.argmax(axis=1)]
        indices = np.stack((np.concatenate((np.arange(x2y_ind.shape[0]), bwd_best)),
                            np.concatenate((fwd_best, np.arange(y2x_ind.shape[0])))), axis=1)
        scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
        pickle.dump(scores, open(args.output+'/scores.bin', 'wb'))
        pickle.dump(indices, open(args.output+'/indices.bin', 'wb'))

    x_sentences = load_text(x_sents_f, stdout=stdout)
    y_sentences = load_text(y_sents_f, stdout=stdout)

    threshold = args.threshold
    min_count = args.min_count
    seen_src, seen_trg = set(), set()
    directory = args.output
    call(f"mkdir -p {directory}")
    src_out = open(
        f'{directory}/all.{args.src_lang}',
        mode='w',
        encoding='utf-8',
        errors='surrogateescape')
    tgt_out = open(
        f'{directory}/all.{args.tgt_lang}',
        mode='w',
        encoding='utf-8',
        errors='surrogateescape')
    scores_out = open(
        f'{directory}/all.scores',
        mode='w',
        encoding='utf-8',
        errors='surrogateescape')
    count = 0
    for i in np.argsort(-scores):
        src_ind, trg_ind = indices[i]
        if not src_ind in seen_src and not trg_ind in seen_trg:
            seen_src.add(src_ind)
            seen_trg.add(trg_ind)
            if scores[i] > threshold or count < min_count:
                print(scores[i], file=scores_out)
                print(x_sentences[src_ind], file=src_out)
                print(y_sentences[trg_ind], file=tgt_out)
                count += 1
    src_out.close()
    tgt_out.close()
    scores_out.close()

    pprint(f"Found {count} pairs for threshold={threshold}", stdout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mine bitext')
    parser.add_argument('--src-lang', help='Source language')
    parser.add_argument('--tgt-lang', help='Target language')
    parser.add_argument('--dim', type=int, default=1024,
        help='Embedding dimension')
    parser.add_argument('--src-dir', help='Source directory')
    parser.add_argument('--tgt-dir', help='Target directory')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--stdout', help='stdout path')
    parser.add_argument('--num-shards', type=int, default=None)
    parser.add_argument('--shard-id', type=int, default=None)
    parser.add_argument('--neighborhood', type=int, default=4,
        help='Embedding dimension')
    parser.add_argument('--threshold', type=float, default=1.06,
        help='Threshold on mined bitext')
    parser.add_argument('--max-sentences', type=int, default=20000000,
        help='Max num sentences used for each language')
    parser.add_argument('--min-count', type=int, default=50000,
        help='Min num sentences used for each language')
    args = parser.parse_args()
    main(args)
