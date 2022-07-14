#!/bin/bash


MODEL_LANGS="afr,amh,ara,asm,ast,aym,azj,bel,ben,bos,bul,cat,ceb,ces,ckb,cym,dan,deu,dyu,ell,eng,est,fas,fin,fra,ful,gla,gle,glg,guj,hat,hau,heb,hin,hrv,hun,hye,ibo,ilo,ind,isl,ita,jav,jpn,kac,kam,kan,kat,kaz,kea,khm,kir,kmb,kon,kor,kur,lao,lav,lin,lit,ltz,lug,luo,mal,mar,mkd,mlg,mlt,mon,mri,msa,mya,nld,nob,npi,nso,nya,oci,orm,ory,pan,pol,por,pus,que,ron,rus,sin,slk,slv,sna,snd,som,spa,sqi,srp,ssw,sun,swe,swh,tam,tel,tgk,tgl,tha,tir,tsn,tur,ukr,umb,urd,uzb,vie,wol,xho,yid,yor,yue,zho_Hans,zul"

LANGS="deu fra afr spa ita dan swe nld dan"
LANGS="fra deu afr spa ita dan swe nld dan"
MOSES=/private/home/schwenk/tools/mosesdecoder/
NORM_PUNCT=$MOSES/scripts/tokenizer/normalize-punctuation.perl
DATADIR=/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/

src_dict="/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/sentencepiece.source.256000.source.dict.txt"
tgt_dict="/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/sentencepiece.source.256000.source.dict.txt"
output_dir="./data/"


SPM=/private/home/costajussa/sentencepiece/build/src/
source=$1
target="eng"
test_pref=$output_dir/devtest$source


  
cat $test_pref.$source | perl ${NORM_PUNCT} -l en | ${SPM}/spm_encode --model=/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/vocab_bin/sentencepiece.source.256000.model --output_format=piece > $test_pref.spm.$source

cat $test_pref.$target | perl ${NORM_PUNCT} -l en | ${SPM}/spm_encode --model=/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/vocab_bin/sentencepiece.source.256000.model --output_format=piece > $test_pref.spm.$target


fairseq-preprocess --source-lang ${source} --target-lang ${target} --testpref $test_pref.spm --srcdict ${src_dict} --tgtdict ${tgt_dict} --destdir ${output_dir} > ${output_dir}/preprocess.${source}-${target}.log


    mv ./data/test.$source-$target.$source.bin $test_pref.$source-$target.$source.bin
    mv ./data/test.$source-$target.$source.idx $test_pref.$source-$target.$source.idx

        mv ./data/test.$source-$target.$target.bin $test_pref.$source-$target.$target.bin
    mv ./data/test.$source-$target.$target.idx $test_pref.$source-$target.$target.idx
    
    cp ${src_dict} ./data/dict.$source.txt
    cp ${tgt_dict} ./data/dict.$target.txt

    cp $test_pref.spm.$source $test_pref.$source-$target.$source
    cp $test_pref.spm.$target $test_pref.$source-$target.$target
   
 
    fairseq-generate ./data/ \
  --path /large_experiments/nllb/mmt/h2_21_models/flores125_v3.3/en_to_many_to_en/v3.3_dense_hrft004.mfp16.mu100000.uf4.lss.enttgt.tmp1.0.shem.NBF.warmup8000.lr0.004.drop0.0.maxtok2560.seed2.valevery200000.max_pos512.adam16bit.fully_sharded.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.ATTDRP0.1.RELDRP0.0.ngpu128/checkpoint_15_100000_consolidated.pt \
        --task=translation_multi_simple_epoch \
        --langs ${MODEL_LANGS} \
        --lang-pairs $source-$target \
        --source-lang $source --target-lang $target \
        --encoder-langtok "tgt" --decoder-langtok \
        --gen-subset devtest$source \
        --skip-invalid-size-inputs-valid-test \
	--beam 4 \
	--max-tokens 512 \
	--bpe "sentencepiece" \
	--sentencepiece-model ${DATADIR}/vocab_bin/sentencepiece.source.256000.model > tmpout

    grep ^D tmpout | cut -f3 > ./data/translation-devtest$source.$source-$target.$target
    

    
      

