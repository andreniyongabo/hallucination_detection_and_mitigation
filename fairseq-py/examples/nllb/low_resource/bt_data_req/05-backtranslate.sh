#!/bin/bash

LANGS=(asm zul amh tel urd vie ita fra deu)
MONO_SIZES=(150k 300k)
SIZES=(1k 5k 10k 50k 25k)

DATA_DIR=/checkpoint/jeanm/nllb/bt_data_req
CHKPT_DIR=/checkpoint/jeanm/nllb/bt_data_req/checkpoints

GENERATE=~/src/fairseq-py/fairseq_cli/generate.py

PARTITION=devaccel

for lang in "${LANGS[@]}"; do
  langs="eng,${lang}"
  lang_pairs="${lang}-eng"
  for size in "${SIZES[@]}"; do
    for mono_size in "${MONO_SIZES[@]}"; do
      data=$DATA_DIR/data-bin/monolingual/$lang.${mono_size}
      echo Backtranslating $lang.$mono_size with best $lang-eng.$size seed model...
      echo "python $GENERATE --fp16 $data \
        --path $CHKPT_DIR/$lang-eng.${size}/checkpoint_best.pt \
        --task=translation_multi_simple_epoch \
        --langs $langs \
        --lang-pairs ${lang_pairs} \
        --source-lang ${lang} --target-lang eng \
        --encoder-langtok "src" \
        --decoder-langtok \
        --gen-subset test \
        --max-tokens 6000 \
        --skip-invalid-size-inputs-valid-test \
        --beam 1 --sampling \
        --num-workers 16" > $data/bt.sh
      sbatch --output $data/backtranslated.$size.out --error $data/backtranslated.$size.err --job-name bt-$lang.${mono_size}-seed$size \
        --gpus-per-node 4 --nodes 1 --cpus-per-task 16 --time 1000 --mem 480G -C volta32gb \
        --partition $PARTITION --ntasks-per-node 1 --open-mode append --no-requeue \
        --wrap "srun sh $data/bt.sh"
      done
  done
done
