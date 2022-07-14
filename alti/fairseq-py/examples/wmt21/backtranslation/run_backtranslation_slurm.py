import argparse
import os
import random
import subprocess
import sacremoses


from glob import glob

random.seed(50)


BUFFER_SIZE = 1024
BATCH_SIZE = 16


MODELS = {
    'en-ha': "/checkpoint/chau/wmt21/bilingual/en_ha.wmt_fb.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.seed1.c10d.det.mt3000.transformer.ELS4.DLS4.encffnx2048.decffnx2048.E512.H8.NBF.ATTDRP0.3.RELDRP0.3.ngpu32/checkpoint40.pt:/checkpoint/chau/wmt21/bilingual/en_ha.wmt_fb.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.seed2.c10d.det.mt3000.transformer.ELS4.DLS4.encffnx2048.decffnx2048.E512.H8.NBF.ATTDRP0.3.RELDRP0.3.ngpu32/checkpoint40.pt:/checkpoint/chau/wmt21/bilingual/en_ha.wmt_fb.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.seed1.c10d.det.mt3000.transformer.ELS4.DLS4.encffnx2048.decffnx2048.E512.H8.NBF.ATTDRP0.3.RELDRP0.3.ngpu32/checkpoint40.pt",
    'ha-en': '/checkpoint/chau/wmt21/bilingual/x_en.ft.joined.32k/ha_en_seed1.pt:/checkpoint/chau/wmt21/bilingual/x_en.ft.joined.32k/ha_en_seed1.pt:/checkpoint/chau/wmt21/bilingual/x_en.ft.joined.32k/ha_en_seed1.pt'
}

DATADIRS = {
    'en-ha': '/private/home/chau/wmt21/multilingual_bin/bilingual_en_x/en_ha.wmt_fb.joined.32k',
    'ha-en': '/private/home/chau/wmt21/multilingual_bin/bilingual/ha_en.wmt_fb.joined.32k'
}


def backtranslate_on_slurm(model, datadir, input_file, out_dir, src, tgt, args):
    input_dir = input_file.split('/')[-2]
    input_shard = input_file.split('/')[-1]
    input_name = input_dir + '/' + input_shard
    if not os.path.exists(os.path.join(out_dir, input_dir)):
        os.mkdir(os.path.join(out_dir, input_dir))
    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)

    spm = os.path.join(datadir, 'sentencepiece.32000.model')
    log_file = os.path.join(out_dir, input_name + '.gen.log')
    output_src = os.path.join(out_dir, input_name + f'.{tgt}-{src}.{tgt}')
    output_tgt = os.path.join(out_dir, input_name + f'.{tgt}-{src}.{src}')
    job_name='bt.'+input_name.replace('/','.')

    job_file_name = os.path.join(slurmdir, f'{job_name}.job')
    if src == 'en':
        langs = f"en,{tgt}"
    else:
        langs = f"en,{src}"
    with open(job_file_name, 'w') as fout:
        fout.write(
f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_file_name}.out
#SBATCH --error={job_file_name}.err

#SBATCH --partition=dev,learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH --time=1320
#SBATCH --constraint=volta32gb
""".format(job_name, slurmdir)
        )

        fout.write(
f"""


cat {input_file} |  python fairseq_cli/interactive.py {datadir} \\
--path {model} \\
--task translation_multi_simple_epoch \\
--langs {langs} \\
--lang-pairs {src}-{tgt} \\
--bpe 'sentencepiece' \\
--sentencepiece-model {spm} \\
--buffer-size {BUFFER_SIZE} \\
--batch-size {BATCH_SIZE} \\
-s {src} -t {tgt} \\
--decoder-langtok \\
--encoder-langtok src \\
--beam 5 \\
--fp16 > {log_file}
cat {log_file} | grep -P "^D-" | cut -f3  > {output_src}
cat {input_file} | cut -d " " -f3- > {output_tgt}
"""
        )

    subprocess.call(['sbatch', job_file_name])

def preprocess_on_slurm(input_file, lang, args):
    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)
    job_name=input_file.split('/')[-1]

    job_file_name = os.path.join(slurmdir, f'{job_name}.job')
    with open(job_file_name, 'w') as fout:
        fout.write(
f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_file_name}.out
#SBATCH --error={job_file_name}.err

#SBATCH --partition=dev,learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=0
#SBATCH --mem=10g
#SBATCH --time=1320
""".format(job_name, slurmdir)
        )

        fout.write(
f"""


MOSES=~edunov/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
cat {input_file} |$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l {lang} | sed -e 's/^/mineddata otherdomain /' > {input_file}.preprocessed
rm {input_file}
"""
        )

    subprocess.call(['sbatch', job_file_name])

def preprocess(input_file, out_dir, lang, args):
    input_name = input_file.split('/')[-1]
    source_prefix = os.path.join(out_dir, input_name)
    if not os.path.exists(source_prefix):
        os.mkdir(source_prefix)

    with open(input_file) as input_f:
        current_size = 0
        i = 0
        current_file = None
        current_f = None

        for line in input_f:
            if current_file is None:
                current_file = f"{source_prefix}/{i}"
                current_f = open(current_file, 'w')
            if current_size >= args.max_batch:
                current_f.close()
                preprocess_on_slurm(current_file, lang, args)
                i += 1
                current_file = None
                current_f = None
                current_size = 0
            else:
                if len(line.split()) > 2 and len(line.split()) < 512 and len(line) < 1024:
                    current_size += 1
                    line = line.strip()
                    current_f.write(line.strip()+'\n')
    current_f.close()
    preprocess_on_slurm(current_file, lang, args)

def main(args):
    src, tgt = args.direction.split('-')
    model = MODELS[args.direction]
    datadir = DATADIRS[args.direction]
    direction_out_dir = os.path.join(args.output_dir, f"{tgt}-{src}")
    lang_preprocessed_dir = os.path.join(args.preprocessed_dir, f"{src}")
    if not os.path.exists(direction_out_dir):
        os.mkdir(direction_out_dir)
    if not os.path.exists(lang_preprocessed_dir):
        os.mkdir(lang_preprocessed_dir)
    input_files = glob(args.paths)
    for input_file in input_files:
        input_name = input_file.split('/')[-1]
        if input_name  == 'cc100_xl.txt' or input_name == 'cc100.txt':
            continue
        if args.mode == 'preprocess':
            preprocess(input_file, lang_preprocessed_dir, src, args)
        elif args.mode == 'backtranslate':
            backtranslate_on_slurm(model, datadir, input_file, direction_out_dir, src, tgt, args)
        else:
            print("Unrecognized mode. Choose preprocess|backtranslate")



"""
Preprocess mode: Split data into small chunks, apply punctuation normalization, filter by length, add domain tag
Backtranslate mode: Run generation, remove domain tag from target side
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths',
        default='/large_experiments/mmt/wmt21/monolingual_preprocessed/en/news.2020.en.shuffled.deduped.0.preprocessed')
            #default='/private/home/angelafan/wmt21/wmt20/monolingual/collated/en_XX/news.2019.en.shuffled.deduped')

    parser.add_argument('--mode')
    parser.add_argument('--direction',
            default='en-ha')
    parser.add_argument('--preprocessed-dir',
            default=f'/large_experiments/mmt/wmt21/monolingual_preprocessed')
    parser.add_argument('--output-dir',
            default=f'/large_experiments/mmt/wmt21/backtranslation')
    parser.add_argument('--processes', type=int, default=64)
    parser.add_argument('--slurmdir', default='slurmdir')
    parser.add_argument('--max-batch', type=int, default=100000)
    main(parser.parse_args())

