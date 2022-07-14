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
DESTDIR = "/large_experiments/mmt/wmt21/bilingual_bin/self_training"
MOSES= "/private/home/edunov/mosesdecoder"
REPLACE_UNICODE_PUNCT = f"{MOSES}/scripts/tokenizer/replace-unicode-punctuation.perl"
NORM_PUNC= f"{MOSES}/scripts/tokenizer/normalize-punctuation.perl"

WMT_ONLY = [
#    "en-ha",
#    "en-is",
    "en-ja",
    "en-ps",
    "en-km",
    "en-ta",
    "en-cs",
    "en-ru",
    "en-zh",
    "en-de",
    "en-pl",
]

WMT_NO_CJK = [
#    "en-ha",
#    "en-is",
    "en-ps",
    "en-km",
    "en-ta",
    "en-cs",
    "en-ru",
    "en-de",
    "en-pl",
]


CJK = [
    "en-zh",
    "en-ja",
    "en-ko",
]

WMT_PLUS = [
#    "en-ha",
    "en-sw",

#    "en-is",
    "en-da",
    "en-no",
    "en-sv",

    "en-ja",
    "en-ko",

    "en-ps",
    "en-fa",
    "en-ur",

    "en-km",
    "en-vi",
    "en-th",
    "en-lo",

    "en-ta",
    "en-ml",
    "en-te",

    # High resource languages
    "en-cs",
    "en-ru",
    "en-zh",
    "en-de",
    "en-pl",
]

domain_tag_dict = {
    "nonfb": "mineddata otherdomain ",
    "all_valid": "wmtdata newsdomain ",
    "parice": "wmtdata newsdomain ",
    "khamenei": "wmtdata newsdomain ",
    "opus": "wmtdata otherdomain ",
    "europarl": "wmtdata otherdomain ",
    "paracrawl": "wmtdata otherdomain ",
    "commoncrawl": "wmtdata otherdomain ",
    "news": "wmtdata newsdomain ",
    "czeng": "wmtdata otherdomain ",
    "yandex": "wmtdata otherdomain ",
    "wikititles": "wmtdata otherdomain ",
    "uncorpus": "wmtdata otherdomain ",
    "tilde": "wmtdata otherdomain ",
    "cwmt": "wmtdata otherdomain ",
    "wikimatrix": "mineddata otherdomain ",
    "jesc": "wmtdata otherdomain ",
    "kftt": "wmtdata otherdomain ",
    "ted": "wmtdata otherdomain ",
    "nunavut": "wmtdata otherdomain ",
    "pmindia": "wmtdata newsdomain ",
    "tanzil": "wmtdata newsdomain ",
    "cvit": "wmtdata otherdomain ",
    "ufal": "wmtdata otherdomain ",
    "bible-ps": "wmtdata otherdomain ",
    "GNOME-ps": "wmtdata otherdomain ",
    "KDE4-ps": "wmtdata otherdomain ",
    "Tatoeba-ps": "wmtdata otherdomain ",
    "ted-wmt20-ps": "wmtdata otherdomain ",
    "Ubuntu-ps": "wmtdata otherdomain ",
    "wikimedia-ps": "wmtdata otherdomain ",
    "GNOME-km": "wmtdata otherdomain ",
    "KDE4-km": "wmtdata otherdomain ",
    "Tatoeba-km": "wmtdata otherdomain ",
    "Ubuntu-km": "wmtdata otherdomain ",
    "GlobalVoices-km": "wmtdata otherdomain ",
    "mined": "mineddata otherdomain ",
}


def pprint(s, stdout=None):
    if stdout is not None:
        print(f"[{datetime.datetime.now()}] {s}", file=stdout)
        stdout.flush()
    print(f"[{datetime.datetime.now()}] {s}")


def execute_in_shell(command):
    # ok guys let's not judge, i code even more slowly in bash than in python
    pprint(command)
    subprocess.call(command, shell=True)


def download_valid(source, target, destdir):
    if target == 'en':
        # X-En
        direction_name = f"en-{source}"
        src_name = "ref"
        tgt_name = "src"
    else:
        # En-X
        direction_name = f"en-{target}"
        src_name = "src"
        tgt_name = "ref"
    #TODO Add ha/is once released
    if source in ["iu", "ja", "pl", "ta"] or target in ["iu", "ja", "pl", "ta"]:
        execute_in_shell(f"python {SACREBLEU} -t wmt20/dev -l {direction_name} --echo {src_name} > {destdir}/valid_temp.{source}_{target}.{source}")
        execute_in_shell(f"python {SACREBLEU} -t wmt20/dev -l {direction_name} --echo {tgt_name} > {destdir}/valid.{source}_{target}.{target}")
    elif source in ["ps", "km"]:
        execute_in_shell(f"cp /private/home/angelafan/wmt21/wmt20/dev_ps_km/wikipedia.dev.{source}-en.{source} {destdir}/valid_temp.{source}_{target}.{source}")
        execute_in_shell(f"cp /private/home/angelafan/wmt21/wmt20/dev_ps_km/wikipedia.dev.{source}-en.en {destdir}/valid.{source}_{target}.{target}")
    elif target in ["ps", "km"]:
        execute_in_shell(f"cp /private/home/angelafan/wmt21/wmt20/dev_ps_km/wikipedia.dev.{target}-en.{target} {destdir}/valid.{source}_{target}.{target}")
        execute_in_shell(f"cp /private/home/angelafan/wmt21/wmt20/dev_ps_km/wikipedia.dev.{target}-en.{source} {destdir}/valid_temp.{source}_{target}.{source}")
    elif target in ["sw", "ig", "no", "sv", "ko", "fa", "ur", "vi", "th", "lo", "ml", "te", "ha", "is"]:
        execute_in_shell(f"cp /private/home/chau/wmt21/non_wmt_valid_data/valid.{direction_name}.{source} {destdir}/valid_temp.{source}_{target}.{source}")
        execute_in_shell(f"cp /private/home/chau/wmt21/non_wmt_valid_data/valid.{direction_name}.{target} {destdir}/valid.{source}_{target}.{target}")
    elif source in ["sw", "ig", "no", "sv", "ko", "fa", "ur", "vi", "th", "lo", "ml", "te", "ha", "is"]:
        execute_in_shell(f"cp /private/home/chau/wmt21/non_wmt_valid_data/valid.{direction_name}.{source} {destdir}/valid_temp.{source}_{target}.{source}")
        execute_in_shell(f"cp /private/home/chau/wmt21/non_wmt_valid_data/valid.{direction_name}.{target} {destdir}/valid.{source}_{target}.{target}")
    else:
        execute_in_shell(f"python {SACREBLEU} -t wmt19 -l {direction_name} --echo {src_name} > {destdir}/valid_temp.{source}_{target}.{source}")
        execute_in_shell(f"python {SACREBLEU} -t wmt19 -l {direction_name} --echo {tgt_name} > {destdir}/valid.{source}_{target}.{target}")
    execute_in_shell(f"sed -e 's/^/wmtdata newsdomain /' {destdir}/valid_temp.{source}_{target}.{source} | {REPLACE_UNICODE_PUNCT} | {NORM_PUNC} -l {source} > {destdir}/valid.{source}_{target}.{source}")
    execute_in_shell(f"rm {destdir}/valid_temp.{source}_{target}.*")
    return [
        f"{destdir}/valid.{source}_{target}.{source}",
        f"{destdir}/valid.{source}_{target}.{target}",
    ]


def download_test(source, target, destdir):
    execute_in_shell(f"python {SACREBLEU} -t wmt20 -l {source}-{target} --echo src > {destdir}/test_temp.{source}_{target}.{source}")
    execute_in_shell(f"python {SACREBLEU} -t wmt20 -l {source}-{target} --echo ref > {destdir}/test.{source}_{target}.{target} ")
    execute_in_shell(f"sed -e 's/^/wmtdata newsdomain /' {destdir}/test_temp.{source}_{target}.{source} | {REPLACE_UNICODE_PUNCT} | {NORM_PUNC} -l {source}> {destdir}/test.{source}_{target}.{source}")
    execute_in_shell(f"rm {destdir}/test_temp.{source}_{target}.*")
    return [
        f"{destdir}/test.{source}_{target}.{source}",
        f"{destdir}/test.{source}_{target}.{target}",
    ]


def train_spm_model(model_prefix, input_data, bpe_size, input_sentence_size=10000000,
        char_coverage=0.99995):
    execute_in_shell(f"""python {SPM_TRAIN} \
        --input={input_data} \
        --model_prefix={model_prefix} \
        --vocab_size={bpe_size} \
        --character_coverage={char_coverage} \
        --input_sentence_size={input_sentence_size} \
        --shuffle_input_sentence=true --model_type=bpe
        """)
    return model_prefix


def apply_spm(spm_model, inputs, outputs):
    execute_in_shell(f"""python {SPM_ENCODE} \
            --model={spm_model} \
            --output_format=piece \
            --inputs={inputs} \
            --processes=60 \
            --outputs={outputs}""")
    return outputs


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


def split_to_shards(prefix, src, tgt, destdir, smallest_shard=250000):
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


def get_corpus_for_vocab(source_files, target_files, destdir, temperature=2.0,
        total_size=10000000):

    # TODO: add temperature sampling
    lang_files = defaultdict(list)
    for direction, fi in source_files.items():
        lang_files[direction.split('-')[0]].append(fi)
    for direction, fi in target_files.items():
        lang_files[direction.split('-')[1]].append(fi)

    lang_counts_map = {
        lang: sum(count_lines(fi) for fi in files)
        for lang, files in lang_files.items()
    }
    sum_count = sum(lang_counts_map.values())
    pprint(f"lang_counts_map: {lang_counts_map}")
    lang_probs_map = {
        lang: count / sum_count for lang, count in lang_counts_map.items()
    }
    lang_probs_map_smoothed = {
        lang: prob ** (1 / temperature) for lang, prob in lang_probs_map.items()
    }
    sum_probs_smoothed = sum(lang_probs_map_smoothed.values())
    total_size = min(total_size, sum_count)
    lang_counts_sampled_map = {
        lang: int(total_size * prob / sum_probs_smoothed)
        for lang, prob in lang_probs_map_smoothed.items()
    }
    pprint(f"Sampled count per lang: {lang_counts_sampled_map}")

    outfile = f"{destdir}/sampled_corpus_vocab"
    with open(outfile, 'w') as out_file:
        for lang, all_corpora in lang_files.items():
            sample_count = lang_counts_sampled_map[lang]
            total_count = lang_counts_map[lang]
            # Get the id's of the sentences that we'll keep
            idx = set(np.random.choice(total_count, sample_count, replace=False))
            i = 0
            for corpus in all_corpora:
                with open(corpus, mode='rb') as in_f:
                    for line in in_f:
                        if i in idx:
                            out_file.write(line.decode())
                        i += 1
            assert (
                total_count == i
            ), f"Number of lines mismatch expected {total_count} vs {i}"

    return outfile


def main(language_list, destdir, bpe_size, use_mined_data=True, use_prod_data=False,
        joined_dict=True, use_wmt=True, use_finetuning=False, use_selftraining=False):
    execute_in_shell(f"mkdir -p {destdir}")
    execute_in_shell(f"mkdir -p {destdir}/sharded_bin")
    train_sources = {}
    train_targets = {}
    for direction in language_list:
        source, target = direction.split('-')
        combined_source_file = f"{destdir}/train.{source}_{target}.{source}"
        combined_target_file = f"{destdir}/train.{source}_{target}.{target}"
        train_sources[direction] = combined_source_file
        train_targets[direction] = combined_target_file
        if os.path.exists(combined_source_file) and os.path.exists(combined_target_file):
            pprint(f"Skipping combining data for {direction}")
            continue
        sorted_direction = "-".join(sorted([source, target]))
        direction_train_sources = []
        if use_wmt:
            direction_train_sources += glob.glob(f'{BITEXT_DIR}/wmt/{sorted_direction}/*.{sorted_direction}.{source}')
            direction_train_sources += glob.glob(f'{BITEXT_DIR}/wmt_additional/{sorted_direction}/*.{sorted_direction}.{source}')
        if use_mined_data:
            direction_train_sources += glob.glob(f'{BITEXT_DIR}/cleaned_mined/{sorted_direction}/*.{sorted_direction}.{source}')
        if use_prod_data:
            direction_train_sources += glob.glob(f'{BITEXT_DIR}/cleaned_non_fb/{sorted_direction}/*.{sorted_direction}.{source}')
        if use_finetuning:
            direction_train_sources += glob.glob(f'/private/home/chau/wmt21/finetuning_data/{sorted_direction}/*.{sorted_direction}.{source}')
        if use_selftraining:
            reverse_direction = f'{target}-{source}'
            direction_train_sources += glob.glob(f'/large_experiments/mmt/wmt21/backtranslation/{reverse_direction}/*.{reverse_direction}.{source}')
        direction_train_targets = [f"{fi[:-2]}{target}" for fi in
                direction_train_sources]

        for fi in direction_train_sources:
            domain = fi.split('.')[0].split('/')[-1]
            if "backtranslation" in fi:
                domain_tag = "mineddata newsdomain "
            elif domain not in domain_tag_dict:
                print(f"Skipping {fi}")
                domain_tag = "mineddata otherdomain "
            else:
                domain_tag = domain_tag_dict[domain]
            execute_in_shell(f"sed -e 's/^/{domain_tag}/' {fi} >> {combined_source_file}")
        for fi in direction_train_targets:
            execute_in_shell(f"cat {fi} >> {combined_target_file}")

    valid_sources = {}
    valid_targets = {}
    test_sources = {}
    test_targets = {}
    for direction in language_list:
        source, target = direction.split('-')
        direction_valids = download_valid(source, target, destdir)
        direction_tests = download_test(source, target, destdir)
        valid_sources[direction] = direction_valids[0]
        valid_targets[direction] = direction_valids[1]
        test_sources[direction] = direction_tests[0]
        test_targets[direction] = direction_tests[1]

    return
    if os.path.exists(f"{destdir}/sentencepiece.{bpe_size}.model") and os.path.exists(f"{destdir}/sentencepiece.{bpe_size}.vocab"):
        pprint("Skipping vocab")
        spm_model = f"{destdir}/sentencepiece.{bpe_size}"
    elif joined_dict:
        vocab_train_corpus = get_corpus_for_vocab(train_sources, train_targets, destdir)
        spm_model = train_spm_model(
            f"{destdir}/sentencepiece.{bpe_size}",
            input_data=vocab_train_corpus,
            bpe_size=bpe_size)
    else:
        assert(len(language_list) == 1)
        direction = language_list[0]

        source, target = direction.split('-')
        source_spm_model = train_spm_model(
            f"{destdir}/sentencepiece.{source}.{bpe_size}",
            input_data=train_sources[direction],
            bpe_size=bpe_size)
        target_spm_model = train_spm_model(
            f"{destdir}/sentencepiece.{target}.{bpe_size}",
            input_data=train_targets[direction],
            bpe_size=bpe_size)
    return


if __name__ == "__main__":
    for direction in [
        'en-ha',
    ]:
        direction_name = direction.replace('-', '_')
        destdir=f"{DESTDIR}/{direction_name}.st.128k"
        main([direction], destdir, 128000, use_wmt=False, use_mined_data=False,
                use_finetuning=False, use_prod_data=False, use_selftraining=True)
