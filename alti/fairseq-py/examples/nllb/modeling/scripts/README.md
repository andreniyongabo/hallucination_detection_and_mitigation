# NLLB utility scripts

## filter_bt.py

This command takes BT shard outputs, extracts a parallel corpus and filters it according to the specified flags.

Example command to run locally with default filtering:
```
python filter_bt.py --local-run --directions eng-fuv --strip-orig-tag --spm-decode $SPM_MODEL_FOLDER/sentencepiece.model $BASE_FOLDER
```
Parameters:
* `$BASE_FOLDER` is typically fairseq's `data_bin`. The script will look for BT shard outputs under `$BASE_FOLDER/bt_out/$direction/*_{0..999}.out`, and will place filtered outputs under `$BASE_FOLDER/bt_filtered/$direction/bt.$lang.gz`.
* `--spm-decode` tells the model to perform SPM decoding using the provided model on both the original and backtranslated text.
* `--directions` is a space-separated list of language pairs. They are interpreted as `$bt_lang-$original_lang` (`$original_lang` corresponds to `S-*` lines in fairseq's output, and `$bt_lang` to `H-*` lines).
* To run on SLURM, remove the `--local-run` flag. You may additionally want to specify `--slurm-partition` and `--slurm-timeout`.
* See `--help` for further information on filtering.


## filter_bt_moses.py

This command is analogous to `filter_bt.py` above, but only performs MOSES-specific filtering (based on the ratio of copied tokens).

Example command to run locally with default filtering:
```
python filter_bt_moses.py  --local-fun --directions eng-fuv --detruecase $MOSES_FOLDER/scripts/recaser/detruecase.perl --spm-decode $SPM_MODEL_FOLDER/sentencepiece.modela $BASE_FOLDER
```

Parameters:
* `$BASE_FOLDER` is the location of input and output shards. The script expects them in `$BASE_FOLDER/{src_sharded,tgt_sharded}/$direction/$lang.{000..999}`. Filtered outputs will be placed in `$BASE_FOLDER/corpora/$direction/bt.$lang.gz`.
* `--detrucase` tells the script to perform detruecasing on both the original and backtranslated shards. You will need to pass the location of the MOSES `detruecase.perl` script.
* `--directions` is interpreted as in `filter_bt.py`.
* To run on SLURM, remove the `--local-run` flag. You may additionally want to specify `--slurm-partition` and `--slurm-timeout`.
* See `--help` for further information on filtering.
