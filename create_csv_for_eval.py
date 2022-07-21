import os
import pandas as pd
from utils import txtToList, saveCSV

DATADIR = "/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k/retrieved_data"
SAVEDDATADIR = "/private/home/andreniyongabo/hallucination_detection_and_mitigation/translations/flores_test"

src = "eng"
tgt = "kin"

src_file = f"{DATADIR}/test.{src}-{tgt}.{src}"
tgt_file = f"{SAVEDDATADIR}/{src}-{tgt}/output.hyp"
sent_score_file = f"{SAVEDDATADIR}/{src}-{tgt}/output.sent_score"
laser_score_file = f"{SAVEDDATADIR}/{src}-{tgt}/output.laser_score"
alti_score_file = f"{SAVEDDATADIR}/{src}-{tgt}/output_translated.alti_score"
annotations_file = f"{SAVEDDATADIR}/{src}-{tgt}/output_first_100.annotation" # Annotations for the first 100 flores devtest pairs translated by nllb200 dense dae model
out_file = f"{SAVEDDATADIR}/{src}-{tgt}/data_for_eval_100.csv"

src_sents = txtToList(src_file)
tgt_sents = txtToList(tgt_file)
sent_scores = txtToList(sent_score_file)
laser_scores = txtToList(laser_score_file)
alti_scores = txtToList(alti_score_file)
annotations = txtToList(annotations_file)

df = pd.DataFrame({src:src_sents[0:100], tgt:tgt_sents[0:100], "sent_score":sent_scores[0:100], "sim_score":laser_scores[0:100], "alti_score":alti_scores[0:100], "annotation":annotations}) # add annotations column if they are available

saveCSV(df, out_file)
