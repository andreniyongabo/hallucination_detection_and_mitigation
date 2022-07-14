#!/bin/bash

DATABINDIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k/
DATADIR=$HOME/hallucination_detection_and_mitigation/translations/flores_test/
SPM=/private/home/costajussa/sentencepiece/build/src/
NORM_PUNCT=/private/home/vedanuj/workspace/fairseq-py/examples/nllb/modeling/preprocessing/moses/normalize-punctuation.perl

src="eng"
tgt="kin"

mkdir $DATADIR/alti_data/
mkdir $DATADIR/alti_data/${src}-${tgt}

data_path=$DATADIR/alti_data/${src}-${tgt}
file_spec="test"
save_path=$DATADIR/${src}-${tgt}/output

# prepare the input
cat ${DATABINDIR}/retrieved_data/test.${src}-${tgt}.${src} | perl ${NORM_PUNCT} -l ${src:0:2} | ${SPM}/spm_encode --model=${DATABINDIR}/vocab_bin/sentencepiece.source.256000.model --output_format=piece > ${data_path}/${file_spec}.${src}-${tgt}.${src}
cat ${DATADIR}/${src}-${tgt}/output.hyp | perl ${NORM_PUNCT} -l ${tgt} | ${SPM}/spm_encode --model=${DATABINDIR}/vocab_bin/sentencepiece.source.256000.model --output_format=piece > ${data_path}/${file_spec}.${src}-${tgt}.${tgt}

cp ${DATABINDIR}/data_bin/shard000/test.${src}-${tgt}.${src}.idx ${data_path}/${file_spec}.${src}-${tgt}.${src}.idx
cp ${DATABINDIR}/data_bin/shard000/test.${src}-${tgt}.${src}.bin ${data_path}/${file_spec}.${src}-${tgt}.${src}.bin

cp ${data_path}/${file_spec}.${src}-${tgt}.${src} ${data_path}/${file_spec}.${src}
cp ${data_path}/${file_spec}.${src}-${tgt}.${tgt} ${data_path}/${file_spec}.${tgt}

cp ${DATABINDIR}/retrieved_data/test.${src}-${tgt}.${src} ${data_path}/${file_spec}-detok.${src}
cp ${DATADIR}/${src}-${tgt}/output.hyp ${data_path}/${file_spec}-detok.${tgt}

cp ${data_path}/${file_spec}.${src}-${tgt}.${src}.idx ${data_path}/${file_spec}_handled.${src}-${tgt}.${src}.idx
cp ${data_path}/${file_spec}.${src}-${tgt}.${src}.bin ${data_path}/${file_spec}_handled.${src}-${tgt}.${src}.bin
python handle_alti_len_issue.py ${data_path}/${file_spec}.${src}-${tgt}.${src} ${data_path}/${file_spec}_handled.${src}-${tgt}.${src}
python handle_alti_len_issue.py ${data_path}/${file_spec}.${src}-${tgt}.${tgt} ${data_path}/${file_spec}_handled.${src}-${tgt}.${tgt}
python handle_alti_len_issue.py ${data_path}/${file_spec}.${src} ${data_path}/${file_spec}_handled.${src}
python handle_alti_len_issue.py ${data_path}/${file_spec}.${tgt} ${data_path}/${file_spec}_handled.${tgt}

wc ${data_path}/${file_spec}_handled.${src}-${tgt}.${src}
wc ${data_path}/${file_spec}_handled.${src}-${tgt}.${tgt}
wc ${data_path}/${file_spec}_handled.${src}
wc ${data_path}/${file_spec}_handled.${tgt}

python $HOME/hallucination_detection_and_mitigation/alti/src/source_contribution_beam.py ${src} ${tgt} ${data_path}/ ${file_spec}_handled ${save_path}
