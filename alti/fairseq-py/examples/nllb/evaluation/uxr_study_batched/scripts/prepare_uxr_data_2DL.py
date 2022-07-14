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

## CC100 2 letter language code used in the M2M-100 model
src = "en"
tgt = "ceb"

# STEP 1: Apply SPM to the raw test/valid data on the source side and target side
# we do this for the dev and devtest splits
command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={OUTDIR}/file.eng --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{tgt}"
print(subprocess.check_output(command, shell=True).decode())
command = f"python scripts/spm_encode.py --model {SPM_MODEL} --output_format=piece   --inputs={OUTDIR}/file.eng --outputs={OUTDIR}/spm_applied/spm.dev.{src}-{tgt}.{src}"
print(subprocess.check_output(command, shell=True).decode())

# STEP 2: Binarized the data
command = f"python fairseq_cli/preprocess.py --source-lang {src} --target-lang {tgt} --validpref {OUTDIR}/spm_applied/spm.dev.{src}-{tgt} --thresholdsrc 0 --thresholdtgt 0 --destdir {OUTDIR}/data_bin --srcdict {DATA_DICT} --tgtdict {DATA_DICT}"
print(subprocess.check_output(command, shell=True).decode())