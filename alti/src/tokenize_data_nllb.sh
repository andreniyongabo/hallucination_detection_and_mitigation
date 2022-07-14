#!/bin/bash

MOSES=/private/home/schwenk/tools/mosesdecoder/
NORM_PUNCT=$MOSES/scripts/tokenizer/normalize-punctuation.perl
DATADIR=/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/
SPM=/private/home/costajussa/sentencepiece/build/src/

src_dict="/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/sentencepiece.source.256000.source.dict.txt"
tgt_dict="/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/sentencepiece.source.256000.source.dict.txt"

src=oci
tgt=eng

for lang in eng oci
do
    cat ./data/devtest.$lang / | perl ${NORM_PUNCT} -l en | ${SPM}/spm_encode --model=/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/vocab_bin/sentencepiece.source.256000.model --output_format=piece > ./data/devtest.spm.$lang
    
done


fairseq-preprocess --only-source --source-lang ${src} --target-lang ${tgt} --testpref ./data/devtest --srcdict ${src_dict} --tgtdict ${tgt_dict} --destdir ./data/data_bin/ > ./data/data_bin/preprocess.${src}-${tgt}.log

