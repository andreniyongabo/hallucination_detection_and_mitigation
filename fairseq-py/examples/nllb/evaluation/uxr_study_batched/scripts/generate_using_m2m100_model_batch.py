# NEED TO RUN ON M2M 12B BRANCH! 

import os
import subprocess

FLORES101_VALID = "/private/home/angelafan/fairseq-py/examples/wmt21/flores_data/flores101_dataset/dev"
FLORES101_TEST = "/private/home/angelafan/fairseq-py/examples/wmt21/flores_data/flores101_dataset/devtest"
SPM_MODEL = "/private/home/shru/12b_gpipe/spm.128k.model"
DATA_DICT = "/private/home/shru/12b_gpipe/data_dict.128k.txt"
MODEL_DICT = "/private/home/shru/12b_gpipe/model_dict.128k.txt"
LANG_PAIRS = "/private/home/shru/12b_gpipe/language_pairs.txt"

# DATADIR = f"/private/home/{os.environ.get('USER')}/uxr_study_data"
DATADIR = f"/private/home/angelafan/uxr_study_data"

GEN_OUTDIR = f"/private/home/{os.environ.get('USER')}/uxr_study_data/m2m100_generations"
os.makedirs(GEN_OUTDIR, exist_ok=True)

LANG3_list = [
    'ara',
    'asm',
#     'azj',
    'bos',
    'bul',
    'ceb',
#     'zho',
    'ces',
    'fin',
    'kat',
#     'deu',
    'hau',
    'isl',
    'jpn',
    'kea',
    'ell',
    'por',
#     'ron',
#     'rus',
    'snd',
    'slv',
    'swh',
    'tel',
    'umb',
    'urd',
    'zul',
]

LANG2_list = [
    'ar',
    'as',
#     'az',
    'bs',
    'bg',
    'ceb', #'cx'
#     'zh',
    'cs',
    'fi',
    'ka',
#     'de',
    'ha',
    'is',
    'ja',
    'q3',
    'el',
    'pt',
#     'ro',
#     'ru',
    'sd',
    'sl',
    'sw',
    'te',
    'qm',
    'ur',
    'zu',
]

for L_key in zip(LANG2_list,LANG3_list):
    LANG2 = L_key[0]
    LANG3 = L_key[1]
    print("Langauge code - 3 letter:",LANG3, "    2 letter:", LANG2)

    ## CC100 2 letter language code used in the M2M-100 model
    src = "en"
    tgt = LANG2

    # ara    ar
    # azj    az
    # kat    ka 
    # jpn    ja
    # por    pt

    # ISO - 3 letter code used in FLORES101
    src_long = "eng"
    tgt_long = LANG3

    generate_command = f"python fairseq_cli/generate.py  \
        /private/home/dlicht/uxr_study_data/data_bin/ \
        --batch-size 1 \
        --path /private/home/shru/projects/12b_last_chk_4_gpus.pt \
        --fixed-dictionary {MODEL_DICT} \
        -s {src} -t {tgt} --remove-bpe 'sentencepiece' --beam 4 \
        --task translation_multi_simple_epoch \
        --lang-pairs {LANG_PAIRS} \
        --decoder-langtok --encoder-langtok src \
        --gen-subset valid \
        --fp16 \
        --dataset-impl mmap \
        --distributed-world-size 1 --distributed-no-spawn \
        --pipeline-model-parallel --pipeline-chunks 1 \
        --pipeline-encoder-balance '[1,15,10]' \
        --pipeline-encoder-devices '[0,1,0]' \
        --pipeline-decoder-balance '[3,11,11,1]'\
        --pipeline-decoder-devices '[0,2,3,0]' \
        --model-overrides \\\"{{'ddp_backend': 'c10d', 'pipeline_balance': '1, 15, 13, 11, 11, 1' , 'pipeline_devices': '0, 1, 0, 2, 3, 0' }}\\\" \
        &> {GEN_OUTDIR}/{src}_{tgt}_beam4.out"
    hyp_tok_cmd = f"cat {GEN_OUTDIR}/{src}_{tgt}_beam4.out | grep ^H | sort -nr -k1.2 | cut -f3-  |  \
        /private/home/edunov/tok.sh {tgt} > {GEN_OUTDIR}/{src}_{tgt}_beam4.hyp"
    tgt_tok_cmd = f"cat {GEN_OUTDIR}/{src}_{tgt}_beam4.out | grep ^T | sort -nr -k1.2 | cut -f2-  | \
        /private/home/edunov/tok.sh {tgt}  > {GEN_OUTDIR}/{src}_{tgt}_beam4.tgt"
    # sacrebleu_cmd = f"cat {GEN_OUTDIR}/{src}_{tgt}_beam4.hyp | \
    #     sacrebleu --tokenize none {GEN_OUTDIR}/{src}_{tgt}_beam4.tgt  | \
    #     tee -a {GEN_OUTDIR}/{src}_{tgt}_beam4.bleu "
    # subprocess.check_output(f"echo \"{generate_command} && {hyp_tok_cmd} && {tgt_tok_cmd} && {sacrebleu_cmd} \" > {GEN_OUTDIR}/{src}_{tgt}_gen.sh", shell=True)
    subprocess.check_output(f"echo \"{generate_command} && {hyp_tok_cmd} && {tgt_tok_cmd} \" > {GEN_OUTDIR}/{src}_{tgt}_gen.sh", shell=True)
    sbatch_command = f"sbatch \
        --output {GEN_OUTDIR}/{src}_{tgt}_eval.out \
        --error {GEN_OUTDIR}/{src}_{tgt}_error.out \
        --job-name {src}-{tgt}.eval \
        --gpus-per-node 8 --nodes 1 --cpus-per-task 80 \
        --time 1000 --mem 480G \
        -C volta32gb \
        --partition learnaccel \
        --ntasks-per-node 1 \
        --open-mode append --no-requeue \
        --wrap \"srun bash {GEN_OUTDIR}/{src}_{tgt}_gen.sh\" "
    print(subprocess.check_output(sbatch_command, shell=True).decode())
