import os
import subprocess

FLORES101_VALID = "/private/home/angelafan/fairseq-py/examples/wmt21/flores_data/flores101_dataset/dev"
FLORES101_TEST = "/private/home/angelafan/fairseq-py/examples/wmt21/flores_data/flores101_dataset/devtest"
SPM_MODEL = "/private/home/shru/12b_gpipe/spm.128k.model"
DATA_DICT = "/private/home/shru/12b_gpipe/data_dict.128k.txt"
MODEL_DICT = "/private/home/shru/12b_gpipe/model_dict.128k.txt"
LANG_PAIRS = "/private/home/shru/12b_gpipe/language_pairs.txt"

OUTDIR = f"/private/home/{os.environ.get('USER')}/uxr_study_data"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/spm_applied", exist_ok=True)

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

    # STEP 1: Apply SPM to the raw test/valid data on the source side and target side
    # we do this for the dev and devtest splits
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={OUTDIR}/file.eng --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{tgt}"
    print(subprocess.check_output(command, shell=True).decode())
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={OUTDIR}/file.eng --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{src}"
    print(subprocess.check_output(command, shell=True).decode())

    # STEP 2: Binarized the data
    command = f"python fairseq_cli/preprocess.py --source-lang {src} --target-lang {tgt} --validpref {OUTDIR}/spm_applied/spm.dev.{src}-{tgt} --thresholdsrc 0 --thresholdtgt 0 --destdir {OUTDIR}/data_bin --srcdict {DATA_DICT} --tgtdict {DATA_DICT}"
    print(subprocess.check_output(command, shell=True).decode())