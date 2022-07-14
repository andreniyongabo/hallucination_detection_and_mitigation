import argparse
import os
import random
import subprocess
import time

from glob import glob
random.seed(50)


BUFFER_SIZE = 1024
BATCH_SIZE = 16


def main(args):
    src, tgt = args.direction.split('-')

    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    ts = int(time.time())
    job_file_name = os.path.join(slurmdir, f"eval_{args.direction}_{ts}")
    tag_name = args.domain_tag.replace(" ","_")
    if args.sentence_splitted:
        test_sources = glob(f"/private/home/chau/wmt21/all_eval_sets/sentence_splitted/{args.direction}/wmt20_splitted.{src}-{tgt}.{src}")
    else:
        test_sources = glob(f"/private/home/chau/wmt21/all_eval_sets/{args.direction}/{args.test_set}.{src}-{tgt}.{src}")

    if args.model is not None:
        model = args.model
    else:
        model = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    for input_file in test_sources:
        test_name = input_file.split('/')[-1].split('.')[0]
        prefix = os.path.join(args.out_dir, f"{test_name}.{args.checkpoint_name[:-3]}.{tag_name}.lenpen{args.lenpen}.beam{args.beam}")
        if args.sentence_splitted:
            prefix += ".splitted"
        orig_file = input_file.replace("wmt20_splitted", "wmt20")
        ref = orig_file[:-2] + tgt
        with open(job_file_name, "w") as fout:
            fout.write(
f"""#!/bin/bash
#SBATCH --job-name=eval_{args.direction}_{ts}
#SBATCH --output={job_file_name}.out
#SBATCH --error={job_file_name}.err

#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH --time=60
#SBATCH --constraint=volta32gb

MOSES=~edunov/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl

cat {input_file} | sed "s/^/{args.domain_tag} /" | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l {src}  | python fairseq_cli/interactive.py {args.datadir} \
  --path {model} \
  --task translation_multi_simple_epoch \
  --langs {args.langs} \
  --lang-pairs {args.direction} \
  --bpe "sentencepiece" \
  --sentencepiece-model {args.spm} \
  --buffer-size {BUFFER_SIZE} \
  --batch-size {BATCH_SIZE} -s {src} -t {tgt} \
  --decoder-langtok \
  --encoder-langtok src  \
  --beam {args.beam} \
  --lenpen {args.lenpen} \
  --fp16 > {prefix}.log
cat {prefix}.log | grep -P "^D-" | cut -f3 > {prefix}.hyp
""")
            if args.sentence_splitted:
                fout.write(f"""
    python /private/home/chau/wmt21/all_eval_sets/sentence_splitted/stitch.py --splitted-file {input_file} --orig-file {orig_file} --translated-file {prefix}.hyp > {prefix}.stitched.hyp
    sacrebleu {ref} -l {args.direction} < {prefix}.stitched.hyp > {prefix}.results
    """)
            else:
                fout.write(f"""
    sacrebleu {ref} -l {args.direction} < {prefix}.hyp > {prefix}.results
    """)
        if args.local:
            subprocess.call(["bash", job_file_name])
        else:
            subprocess.call(["sbatch", job_file_name])


"""
SPM encode data by launching many jobs on SLURM.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEFAULT_CP = "/checkpoint/chau/wmt21/bitext_bt_v3_bilingual/wmt_only.bitext_bt.v3.16_shards.en-ja.transformer_12_12.fp16.SPL_temperature.tmp5.adam.lr0.001.drop0.1.ls0.1.seed1234.c10d.det.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32"
    DEFAULT_SPM = "/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards/sentencepiece.128000.model"
    DEFAULT_DATADIR = "/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards"
    parser.add_argument("--model",
            default=None,
            help="model arg (could be ensemble), supersedes --checkpoint-dir and --checkpoint-name") 
    parser.add_argument("--checkpoint-dir",
            default=DEFAULT_CP)
    parser.add_argument("--out-dir",
            default=DEFAULT_CP)
    parser.add_argument("--direction",
            default="en-ja")
    parser.add_argument("--checkpoint-name",
            default="checkpoint_best.pt")
    parser.add_argument("--datadir",
            default=DEFAULT_DATADIR)
    parser.add_argument("--langs",
            default="en,ja")
    parser.add_argument("--spm",
            default=DEFAULT_SPM)
    parser.add_argument("--domain-tag", default="wmtdata newsdomain")
    parser.add_argument("--lenpen", default=1.0)
    parser.add_argument("--beam", default=4)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--sentence-splitted", action="store_true")
    parser.add_argument("--test-set", default="*")
    parser.add_argument("--partition", default="learnfair")
    parser.add_argument("--slurmdir", default="eval_slurmdir")
    args = parser.parse_args()
    main(args)
