# Backtranslation data requirements experiments

These experiments aim to investigate the minimum amounts of seed bitext and monolingual
data required for kick-starting an iterative backtranslation pipeline for a
low-resource language.

For a set of low-resource language, bilingual models are trained in the `eng-xxx` and
`xxx-eng` directions using varying amounts of seed bitext.  These are evaluated, and then
the `xxx-eng` models are used to backtranslate monolingual data for each of the low-res
languages. A new set of `eng-xxx` models are then trained with this additional data, and
evaluated again.

Note: some high-resource languages are also included. They're useful to include since
low-resource languages often have untrustworthy LID, and looking at languages with
reliable LID allows us to focus on the backtranslation without having to worry about
external factors.

This backtranslation pipeline is run with the scripts in this directory, which are
numbered for ease of use.

## 1. Create corpora

The script `01a-create-bitext.sh` creates seed corpora from the FLORES training data,
and copies over the FLORES dev and devtest sets. Then, `01b-create-mono-data-from-cc.sh`
and `01c-create-mono-data-from-minimining.sh` create monolingual data by sourcing
it from a few different places. Monolingual datasets are created at a fixed set of
sizes.

# 2. Binarise the data

In `02a-create-dictionaries.sh` a dictionary is created for each language, using the
largest monolingual and seed datasets to build it. The same dictionary is also used
for models trained on smaller seed datasets.

All data is then binarised in `02b-binarise-data.sh`.

# 3. Train the seed models

Seed models for `xxx-eng` and `xxx-eng` are then trained in `03-train-seed-models.py`.

# 4. Evaluate seed models

In `04a-eval-seed-models.sh` we evaluate the trained models. The results are then
collected in `04b-collect-bitext-eval.py`, and symbolic links are created to keep track
of the best performing model for a given seed corpus and direction.

# 5-6. Backtranslation

Using the best model for the `xxx-eng` direction identified in the previous step, in
`05-backtranslate.sh` we backtranslate the data. This data is then binarised and
combined with the seed data in `06-binarise-bt-data.sh`.

# 7. Train on the backtranslated data

A new model is then trained for each direction using the combined seed and augmented
data, using upsampling such that the two types of data are evenly balanced.

# 8. Evaluate the augmented models

Similarly to step 4, the models are then evaluated and results are collected.

