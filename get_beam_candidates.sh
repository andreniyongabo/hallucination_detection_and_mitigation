#!bin/bash
"""
Get all the candidates from beam and their score
"""
CHECKPOINT_DIR=/checkpoint/vedanuj/nmt/flores200/dense_dae_ssl.mfp16.mu100000.uf2.lss.tmp1.0.lr0.001.drop0.1.maxtok5120.seed2.max_pos512.shem.NBF.adam16bit.fully_sharded.enttgt.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E2048.H16.ATTDRP0.1.RELDRP0.0.m0.3.mr0.1.i0.0.p0.0.r0.0.pl3.5.rl1.ngpu128
DATADIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k
MOSES=/private/home/vedanuj/workspace/fairseq-py/examples/nllb/modeling/preprocessing/moses
REPLACE_UNICODE_PUNCT=$MOSES/replace-unicode-punctuation.perl
NORM_PUNCT=$MOSES/normalize-punctuation.perl
CUSTOMDIR=/private/home/andreniyongabo/mytest/translations/flores_test/

langs="ace_Arab,ace_Latn,acm,acq,aeb,afr,ajp,aka,amh,apc,ara_Arab,ars,ary,arz,asm,ast,awa,ayr,azb,azj,bak,bam,ban,bel,bem,ben,bho,bjn_Arab,bjn_Latn,bod,bos,bug,bul,cat,ceb,ces,cjk,ckb,crh_Latn,cym,dan,deu,dik,dyu,dzo,ell,eng,epo,est,eus,ewe,fao,fas,fij,fin,fon,fra,fur,fuv,gla,gle,glg,grn,guj,hat,hau,heb,hin,hne,hrv,hun,hye,ibo,ilo,ind,isl,ita,jav,jpn,kab,kac,kam,kan,kas_Arab,kas_Deva,kat,kau_Arab,kau_Latn,kaz,kbp,kea,khm,kik,kin,kir,kmb,kon,kor,kur,lao,lav,lij,lim,lin,lit,lmo,ltg,ltz,lua,lug,luo,lus,mag,mai,mal,mar,min_Latn,mkd,mlg,mlt,mni_Mtei,mon,mos,mri,msa,mya,nld,nno,nob,npi,nso,nus,nya,oci,orm,ory,pag,pan,pap,pol,por,prs,pus,que,ron,run,rus,sag,san,sat,scn,shn,sin,slk,slv,smo,sna,snd,som,sot,spa,sqi,srd,srp_Cyrl,ssw,sun,swe,swh,szl,tam,tat_Cyrl,tel,tgk,tgl,tha,tir,tmh_Latn,tmh_Tfng,tpi,tsn,tso,tuk,tum,tur,twi,tzm,uig,ukr,umb,urd,uzb,vec,vie,war,wol,xho,yid,yor,yue,zho_Hans,zho_Hant,zul"

src="eng"
tgt="kin"
input_file=${DATADIR}/retrieved_data/test.${src}-${tgt}.${src}
ref_file=${DATADIR}/retrieved_data/test.${src}-${tgt}.${tgt}
mkdir -p ${CUSTOMDIR}/beam_candidates/
mkdir -p ${CUSTOMDIR}/beam_candidates/${src}-${tgt}/
prefix=${CUSTOMDIR}/${src}-${tgt}/output
beam=4
batch_size=1

# for tokenization, if src=="eng" use ${src:0:2}, otherwise use ${src}
cat ${input_file} |  ${NORM_PUNCT} -l ${src:0:2} | python fairseq_cli/interactive_beam_candidates.py ${DATADIR}/data_bin/shard000 \
    --path ${CHECKPOINT_DIR}/checkpoint_4_100000-shard0.pt \
    --task translation_multi_simple_epoch \
    --langs ${langs} \
    --source-lang ${src} \
    --target-lang ${tgt} \
    --lang-pairs ${src}-${tgt} \
    --bpe "sentencepiece" \
    --sentencepiece-model ${DATADIR}/vocab_bin/sentencepiece.source.256000.model \
    --batch-size ${batch_size} \
    --decoder-langtok \
    --encoder-langtok tgt  \
    --beam ${beam} \
    --lenpen 1.0 \
    --fix-batches-to-gpus \
    --fp16  > ${prefix}.gen_log 
