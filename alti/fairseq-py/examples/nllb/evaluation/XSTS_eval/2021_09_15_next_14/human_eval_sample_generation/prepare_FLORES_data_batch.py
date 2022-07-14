import os
import subprocess

FLORES101_VALID = "/private/home/angelafan/fairseq-py/examples/wmt21/flores_data/flores101_dataset/dev"
FLORES101_TEST = "/private/home/angelafan/fairseq-py/examples/wmt21/flores_data/flores101_dataset/devtest"
SPM_MODEL = "/private/home/shru/12b_gpipe/spm.128k.model"
DATA_DICT = "/private/home/shru/12b_gpipe/data_dict.128k.txt"
MODEL_DICT = "/private/home/shru/12b_gpipe/model_dict.128k.txt"
LANG_PAIRS = "/private/home/shru/12b_gpipe/language_pairs.txt"

OUTDIR = f"/private/home/{os.environ.get('USER')}/flores101_data"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/spm_applied", exist_ok=True)

LANG3_list = [
    'rus',
    'deu',
    'hau',
    'isl',
    'ces',
    'jpn',
    'snd',
    'azj',
    'amh',
    'zul',
    'kat',
    'urd',
    'ara',
    'hin',
    'slv',
    'swh',
    'bul',
    'bos',
    'ron',
    'por',
    'zho_simpl',
    'fra',
    'kor',
    'spa',

    'ful',
    'lug',
    'som',
    'mri',
    'luo',
    'pan',
    'nep',

    'lin',
    'kan',
    'nso',
]

LANG2_list = [
    'ru',
    'de',
    'ha',
    'is',
    'cs',
    'ja',
    'sd',
    'az',
    'am',
    'zu',
    'ka',
    'ur',
    'ar',
    'hi',
    'sl',
    'sw',
    'bg',
    'bs',
    'ro',
    'pt',

    'zho',
    'fr',
    'ko',
    'es',

    'ff',
    'lg',
    'so',
    'mi',
    'luo',
    'pa',
    'ne',

    'ln',
    'kn',
    'ns',
]

# run first for into English
for L_key in zip(LANG2_list,LANG3_list):
    LANG2 = L_key[0]
    LANG3 = L_key[1]
    print("Langauge code - 3 letter:",LANG3, "    2 letter:", LANG2)

    ## CC100 2 letter language code used in the M2M-100 model
    src = LANG2
    tgt = "en"

    # ISO - 3 letter code used in FLORES101
    src_long = LANG3
    tgt_long = "eng"

    # STEP 1: Apply SPM to the raw test/valid data on the source side and target side
    # we do this for the dev and devtest splits

    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_VALID}/{tgt_long}.dev --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{tgt}"
    print(subprocess.check_output(command, shell=True).decode())
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_VALID}/{src_long}.dev --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{src}"
    print(subprocess.check_output(command, shell=True).decode())
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_TEST}/{tgt_long}.devtest --outputs={OUTDIR}/spm_applied/spm.devtest.{src}-{tgt}.{tgt}"
    print(subprocess.check_output(command, shell=True).decode())
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_TEST}/{src_long}.devtest --outputs={OUTDIR}/spm_applied/spm.devtest.{src}-{tgt}.{src}"
    print(subprocess.check_output(command, shell=True).decode())

    # STEP 2: Binarized the data
    command = f"python fairseq_cli/preprocess.py --source-lang {src} --target-lang {tgt} --validpref {OUTDIR}/spm_applied/spm.dev.{src}-{tgt} --testpref {OUTDIR}/spm_applied/spm.devtest.{src}-{tgt} --thresholdsrc 0 --thresholdtgt 0 --destdir {OUTDIR}/data_bin --srcdict {DATA_DICT} --tgtdict {DATA_DICT}"
    print(subprocess.check_output(command, shell=True).decode())


# and run again, but with the langauge direction reversed to be out of English
for L_key in zip(LANG2_list,LANG3_list):
    LANG2 = L_key[0]
    LANG3 = L_key[1]
    print("Langauge code - 3 letter:",LANG3, "    2 letter:", LANG2)

    ## CC100 2 letter language code used in the M2M-100 model
    src = "en"
    tgt = LANG2

    # ISO - 3 letter code used in FLORES101
    src_long = "eng"
    tgt_long = LANG3

    # STEP 1: Apply SPM to the raw test/valid data on the source side and target side
    # we do this for the dev and devtest splits

    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_VALID}/{tgt_long}.dev --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{tgt}"
    print(subprocess.check_output(command, shell=True).decode())
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_VALID}/{src_long}.dev --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{src}"
    print(subprocess.check_output(command, shell=True).decode())

    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_TEST}/{tgt_long}.devtest --outputs={OUTDIR}/spm_applied/spm.devtest.{src}-{tgt}.{tgt}"
    print(subprocess.check_output(command, shell=True).decode())
    command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={FLORES101_TEST}/{src_long}.devtest --outputs={OUTDIR}/spm_applied/spm.devtest.{src}-{tgt}.{src}"
    print(subprocess.check_output(command, shell=True).decode())

    # STEP 2: Binarized the data
    command = f"python fairseq_cli/preprocess.py --source-lang {src} --target-lang {tgt} --validpref {OUTDIR}/spm_applied/spm.dev.{src}-{tgt} --testpref {OUTDIR}/spm_applied/spm.devtest.{src}-{tgt} --thresholdsrc 0 --thresholdtgt 0 --destdir {OUTDIR}/data_bin --srcdict {DATA_DICT} --tgtdict {DATA_DICT}"
    print(subprocess.check_output(command, shell=True).decode())