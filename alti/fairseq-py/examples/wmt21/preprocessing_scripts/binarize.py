import argparse
import ipdb
import os
import numpy as np
import re
import gzip
from collections import defaultdict, namedtuple
from itertools import chain
import subprocess
import sacremoses
import glob
import hashlib
import datetime


BITEXT_DIR = "/private/home/angelafan/wmt21/finished_bitexts"
SACREBLEU = "/private/home/angelafan/sacrebleu/sacrebleu/sacrebleu.py"
SPM_TRAIN = "/private/home/angelafan/mmt/fairseq-py/scripts/spm_train.py"
SPM_ENCODE = "examples/wmt21/preprocessing_scripts/spm_encode_multiprocess.py"
CLEAN_CORPUS = "/private/home/angelafan/mmt/mosesdecoder/scripts/training/clean-corpus-n.perl"
PREPROCESS = "/private/home/angelafan/mmt/fairseq-py/preprocess.py"
DESTDIR = "/private/home/chau/wmt21/multilingual_bin"


def pprint(s, stdout=None):
    if stdout is not None:
        print(f"[{datetime.datetime.now()}] {s}", file=stdout)
        stdout.flush()
    print(f"[{datetime.datetime.now()}] {s}")


def execute_in_shell(command):
    # ok guys let's not judge, i code even more slowly in bash than in python
    pprint(command)
    subprocess.call(command, shell=True)
def clean_corpus(inputs, source, target, outputs):
    execute_in_shell(f"perl {CLEAN_CORPUS} -ratio 2.5 {inputs} {source} {target} {outputs} 1 250")


def dedup(
        source_in, target_in,
        source_out, target_out,
        valid_source, valid_target,
        test_source, test_target):
    mem = set()
    with open(valid_source) as valid_s_f, open(valid_target) as valid_t_f, \
            open(test_source) as test_s_f, open(test_target) as test_t_f:
        for source_line, target_line in zip(valid_s_f, valid_t_f):
            concated = f"{source_line} {target_line}".lower()
            hashed = hashlib.md5(concated.encode()).hexdigest()
            mem.add(hashed)
        for source_line, target_line in zip(test_s_f, test_t_f):
            concated = f"{source_line} {target_line}".lower()
            hashed = hashlib.md5(concated.encode()).hexdigest()
            mem.add(hashed)

    filtered_count = 0
    pprint(f"Start dedup")
    with open(source_in) as source_in_f, open(target_in) as target_in_f:
        with open(source_out, 'w') as source_out_f, open(target_out, 'w') as target_out_f:
            for source_line, target_line in zip(source_in_f, target_in_f):
                concated = f"{source_line} {target_line}".lower()
                hashed = hashlib.md5(concated.encode()).hexdigest()
                if hashed in mem or len(source_line.strip()) < 1 or len(target_line.strip()) < 1:
                    filtered_count += 1
                    continue
                mem.add(hashed)
                source_out_f.write(source_line)
                target_out_f.write(target_line)
    pprint(f"Done dedup, filtered {filtered_count} duplicates")


def count_lines(fi):
    count = 0
    for line in open(fi):
        count += 1
    return count


def split_to_shards(prefix, src, tgt, destdir, smallest_shard=2000000):
    num_lines = count_lines(f"{prefix}.{src}")
    if num_lines <= smallest_shard:
        num_shards = 1
    elif num_lines <= 2 * smallest_shard:
        num_shards = 2
    elif num_lines <= 4 * smallest_shard:
        num_shards = 4
    elif num_lines <= 8 * smallest_shard:
        num_shards = 8
    elif num_lines <= 16 * smallest_shard:
        num_shards = 16
    elif num_lines <= 32 * smallest_shard:
        num_shards = 32
    else:
        num_shards = 64

    idx_to_shard = np.random.randint(0, num_shards, num_lines)
    idx = 0
    source_output_files = []
    target_output_files = []
    for i in range(num_shards):
        execute_in_shell(f"mkdir -p {destdir}/sharded_bin/shard{i:03d}")
        source_output_files.append(
            open(f"{destdir}/sharded_bin/shard{i:03d}/train.sharded.{src}_{tgt}.{src}" ,"w"))
        target_output_files.append(
            open(f"{destdir}/sharded_bin/shard{i:03d}/train.sharded.{src}_{tgt}.{tgt}" ,"w"))

    with open(f"{prefix}.{src}") as src_f, \
            open(f"{prefix}.{tgt}") as tgt_f:
        for src_line, tgt_line in zip(src_f, tgt_f):
            shard_id = idx_to_shard[idx]
            source_output_files[shard_id].write(src_line)
            target_output_files[shard_id].write(tgt_line)
            idx += 1
    for fi in source_output_files:
        fi.flush()
        fi.close()
    for fi in target_output_files:
        fi.flush()
        fi.close()
    return [f"{destdir}/sharded_bin/shard{i:03d}/train.sharded.{src}_{tgt}" for i in range (num_shards)]


def binarize(
        source, target,
        trainpref, validpref,
        srcdict, tgtdict, destdir):
    execute_in_shell(f"""export MKL_SERVICE_FORCE_INTEL=1 && fairseq-preprocess \
        --source-lang {source} \
        --target-lang {target} \
        --trainpref {trainpref} \
        --validpref {validpref} \
        --srcdict {srcdict} \
        --tgtdict {tgtdict} \
        --destdir {destdir} --workers 60 \
        > {destdir}/preprocess.{source}-{target}.log""")


def main(direction, destdir, src_spm_vocab, tgt_spm_vocab):
    source, target = direction.split('-')
    for split in ['train', 'valid', 'test']:
        for lang in [source, target]:
            if os.path.exists(f"{destdir}/spm_outs/{split}.{source}_{target}.{lang}"):
                execute_in_shell(f"mv {destdir}/spm_outs/{split}.{source}_{target}.{lang} {destdir}/{split}.spm.{source}_{target}.{lang}")

    if not os.path.exists(os.path.join(destdir, f"train.spm.clean.dedup.{source}_{target}.{source}")) \
        or not os.path.exists(os.path.join(destdir, f"train.spm.clean.dedup.{source}_{target}.{target}")):
        clean_corpus(
            os.path.join(destdir, f"train.spm.{source}_{target}"),
            source, target,
            os.path.join(destdir, f"train.spm.clean.{source}_{target}"),
        )
        dedup(
            os.path.join(destdir, f"train.spm.clean.{source}_{target}.{source}"),
            os.path.join(destdir, f"train.spm.clean.{source}_{target}.{target}"),
            os.path.join(destdir, f"train.spm.clean.dedup.{source}_{target}.{source}"),
            os.path.join(destdir, f"train.spm.clean.dedup.{source}_{target}.{target}"),
            os.path.join(destdir, f"valid.spm.{source}_{target}.{source}"),
            os.path.join(destdir, f"valid.spm.{source}_{target}.{target}"),
            os.path.join(destdir, f"test.spm.{source}_{target}.{source}"),
            os.path.join(destdir, f"test.spm.{source}_{target}.{target}"),
        )
    train_shards = split_to_shards(
        os.path.join(destdir, f"train.spm.clean.dedup.{source}_{target}"),
        source, target,
        destdir)
    src_fairseq_dict = f"{destdir}/dict.{source}.txt"
    tgt_fairseq_dict = f"{destdir}/dict.{target}.txt"
    if not os.path.exists(src_fairseq_dict):
        execute_in_shell(f"""tail -n +4 {src_spm_vocab} | awk '{{print $1" "1}}' > {src_fairseq_dict}""")
    if not os.path.exists(tgt_fairseq_dict):
        execute_in_shell(f"""tail -n +4 {tgt_spm_vocab} | awk '{{print $1" "1}}' > {tgt_fairseq_dict}""")
    for shard_id, train_shard in enumerate(train_shards):
        binarize(
            source, target,
            train_shard,
            os.path.join(destdir, f"valid.spm.{source}_{target}"),
            src_fairseq_dict,
            tgt_fairseq_dict,
            f"{destdir}/sharded_bin/shard{shard_id:03d}",
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destdir',
            default='/private/home/chau/wmt21/multilingual_bin/bilingual/en_ha.wmt.separated.32k')
    parser.add_argument('--direction',
            default='en-ha')
    parser.add_argument('--src-spm-vocab',
            default='/private/home/chau/wmt21/multilingual_bin/bilingual/en_ha.wmt.separated.32k/sentencepiece.en.32000.vocab')
    parser.add_argument('--tgt-spm-vocab',
            default='/private/home/chau/wmt21/multilingual_bin/bilingual/en_ha.wmt.separated.32k/sentencepiece.ha.32000.vocab')
    args = parser.parse_args()
    main(args.direction, args.destdir, args.src_spm_vocab, args.tgt_spm_vocab)
