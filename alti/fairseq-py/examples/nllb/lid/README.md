# NLLB LID (Language IDentification)

This subdirectory contains code associated with LID (Language IDentification): training, evaluation, optimization, experimentation scripts.




## Compare Two LID Models

Example:

```
./compare_display_results2.py \
    --old /large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/result/result.classifiermetrics-flores-devtest.fasttext.1.txt \
    --new /large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/result/result.classifiermetrics-flores-devtest.fasttext.7.txt \
    --eval-langs-only

./compare_display_results2.py \
    --old /large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/result/result.classifiermetrics-flores-filled.fasttext.7.txt \
    --new /large_experiments/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/result/result.classifiermetrics-flores-filled.fasttext.7.txt \
```

## Running sanity checks
The `data_sanity` folder contains code for a [simple dashboard in Weights & Biases](https://fairwandb.org/nllb/data-sanity) that checks the quality of LID on a corpus whose language is known beforehand.

You can refresh the dashboard by running
```
python data_sanity_dashboard.py \
    --model /private/home/celebio/lid_latest_models/2022-02-18_ft_model.bin
```