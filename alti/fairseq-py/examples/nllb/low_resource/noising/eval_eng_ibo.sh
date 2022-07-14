#CHECKPOINT_DIR=/checkpoint/chau/noising_augmentation/noising/noising.fp16.uf4.entsrc.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt4000.m0.3.mr0.1.i0.0.p0.0.r0.0.pl3.5.mlspan-poisson.rl1.transformer.ELS6.encffnx4096.E1024.H16.NBF.ATTDRP0.2.RELDRP0.2.ngpu32
CHECKPOINT_DIR=/checkpoint/chau/noising_augmentation/baseline/baseline.fp16.uf4.entsrc.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt4000.transformer.ELS6.encffnx4096.E1024.H16.NBF.ATTDRP0.2.RELDRP0.2.ngpu32
#CHECKPOINT_DIR=/checkpoint/chau/noising_augmentation/baseline/baseline.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt4000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
DATADIR=/large_experiments/mmt/demo/flores2_x_en.32k
mono_data_dir=/large_experiments/mmt/demo/flores2_x_en.32k/data_bin/monolingual
MOSES=~chau/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNCT=$MOSES/scripts/tokenizer/normalize-punctuation.perl
input_file=/large_experiments/mmt/flores101/devtest/eng.devtest
ref_file=/large_experiments/mmt/flores101/devtest/ibo.devtest
src=eng
tgt=ibo
langs="eng,hau,ibo"

mkdir -p ${CHECKPOINT_DIR}/${src}-${tgt}
prefix=${CHECKPOINT_DIR}/${src}-${tgt}/flores_test
cat ${input_file} |  ${NORM_PUNCT} -l en  | python fairseq_cli/interactive.py ${DATADIR}/data_bin/shard000 \
  --path ${CHECKPOINT_DIR}/checkpoint_best.pt \
  --task translation_multi_simple_epoch \
  --langs ${langs} \
  --lang-pairs ${src}-${tgt} \
  --bpe "sentencepiece" \
  --sentencepiece-model ${DATADIR}/vocab_bin/sentencepiece.source.32000.model \
  --buffer-size 1024 \
  --batch-size 16 -s ${src} -t ${tgt} \
  --decoder-langtok \
  --encoder-langtok src  \
  --beam 4 \
  --lenpen 1.0 \
  --fp16  > ${prefix}.gen_log
  #--extra-data "{'mono_dae': '${mono_data_dir}'}" \
  #--extra-lang-pairs "{'mono_dae': 'hau,ibo'}" \
  #--langtoks "{'mono_dae': ('src', 'tgt')}"  \

cat ${prefix}.gen_log | grep -P "^D-" | cut -f3 > ${prefix}.hyp
sacrebleu ${ref_file} -l en-ig < ${prefix}.hyp > ${prefix}.results
