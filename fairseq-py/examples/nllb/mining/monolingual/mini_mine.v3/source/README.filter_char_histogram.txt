Example use:

```
xzcat /large_experiments/nllb/mmt/data/monolingual/raw/srp_Cyrl/xlsum.srp_Cyrl.xz | \
  ./filtering/filter_char_histogram.py \
  --lang srp \
  --threshold 0.8 \
  --histogram-threshold 0.95 \
  --histograms /large_experiments/mmt/lidruns/2021-09-20-14-14-histogram-baseline/histograms/valid/ \
  2> rejectd.srp.txt \
  1> cleaned.srp.txt

```
