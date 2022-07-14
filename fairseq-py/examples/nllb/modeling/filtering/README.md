# Bitext filtering

The `filter.py` pipeline applies various filters to bitext, with optional support for monolingual text. It is configured with [Hydra](https://hydra.cc/).

A basic run using default parameters might look like this (run from the fairseq-py root):
```
python examples/nllb/modeling/filtering/filter.py \
  output_dir=/checkpoint/$USER/filter_test \
  data_conf_dir=`pwd`/examples/nllb/modeling/components_conf/data/
```
This command will run using the output directory and data\_conf directory specified above, and will additionally load the default example config `conf/example.yaml`. Anything not specified on the command line or in `conf/example.yaml` will be set to the default values specified in `data_types.FilterConfig`.

Note, for instance, how in both `conf/example.yaml` and `conf/200_primary.yaml` we have specified `train_primary.lid_filter.excluded_corpora=['fbseed']`, which tells the LID filter not to run on the `fbseed` corpus.

Jobs can be run either locally or on SLURM. Have a look at the `executor` config key in `conf/example.yaml` and `conf/200_primary.yaml`, which run locally and on SLURM respectively.

When needing to run a new filtering job with many parameter overrides, instead of manually overriding parameters on the command line it is better to create an entirely new config file, e.g. `conf/new.yaml`, containing all overrides. The script can then be instructed to load it as follows:
```
python filter.py --config-name=new output_dir=[...] data_conf_dir=[...]
```
