for model_prefix in african26.x_en.64k/checkpoints/african26.x_en.64k.uf4.lr0.001.drp0.1.ls0.1.seed1234.els12.dls12.adrp0.1.rdrp0.0.ngpu64  african26.x_en.64k/checkpoints/african26.x_en.64k.uf4.lr0.001.drp0.2.ls0.1.seed1234.els12.dls12.adrp0.1.rdrp0.0.ngpu64; do
  for src in amh ful dyu hau ibo kam kmb kon lin lug luo nso nya orm sna som ssw swh tir tsn umb wol xho yor zul; do
    tgt=eng
    CHECKPOINT_DIR=/checkpoint/chau/nllb/${model_prefix}

    DATADIR=/large_experiments/mmt/multilingual_bin/flores_african_en_26langs.64k
    MOSES=~chau/mosesdecoder
    REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
    NORM_PUNCT=$MOSES/scripts/tokenizer/normalize-punctuation.perl
    input_file=/large_experiments/mmt/flores101/devtest/${src}.devtest
    ref_file=/large_experiments/mmt/flores101/devtest/${tgt}.devtest
    langs="eng,amh,ful,dyu,hau,ibo,kam,kmb,kon,lin,lug,luo,nso,nya,orm,sna,som,ssw,swh,tir,tsn,umb,wol,xho,yor,zul"

    mkdir -p ${CHECKPOINT_DIR}/${src}-${tgt}
    prefix=${CHECKPOINT_DIR}/${src}-${tgt}/flores_test
    cat ${input_file} |  ${NORM_PUNCT} -l ${src:0:2}  | python fairseq_cli/interactive.py ${DATADIR}/data_bin/shard000 \
      --path ${CHECKPOINT_DIR}/checkpoint_best.pt \
      --task translation_multi_simple_epoch \
      --langs ${langs} \
      --lang-pairs ${src}-${tgt} \
      --bpe "sentencepiece" \
      --sentencepiece-model ${DATADIR}/vocab_bin/sentencepiece.source.64000.model \
      --buffer-size 1024 \
      --batch-size 16 -s ${src} -t ${tgt} \
      --decoder-langtok \
      --encoder-langtok src  \
      --beam 4 \
      --lenpen 1.0 \
      --fp16  > ${prefix}.gen_log

    cat ${prefix}.gen_log | grep -P "^D-" | cut -f3 > ${prefix}.hyp
    sacrebleu ${ref_file} -l ${src:0:2}-${tgt:0:2} < ${prefix}.hyp > ${prefix}.results
  done
done
