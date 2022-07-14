#!/bin/bash




LANGS="afr spa amh ara asm ast azj bel ben bos bul cat ceb ces ckb cym dan deu ell est fas fin fra ful gle glg guj hau heb hib hrv hun hye ibo ind isl ita jav jpn kam kan kat kaz kea khm kir kor lao lav lin lit ltz lug luo mal mar mkd mlt mon mri msa mya nld nob npi nsp nya oci orm ory pan pol por pus ron rus slk slv sna snd som spa srp swe swh tam tel tgk tgl tha tur ukr umb urd uzb vie wol xho yor zul"

LANGS="dyu gla hat hun ilo kac kmb kon kur mlg nso que sin sqi ssw sun tir tsn yue zho_Hans"



#LANGS="afr spa"
#LANGS="nya oci orm ory pan pol por pus ron rus slk slv sna snd som spa srp swe swh tam tel tgk tgl tha tur ukr umb urd uzb vie wol xho yor zul"

for lang in $LANGS
    do
#		./filter-sent.bash $lang eng

		python item_source_contribution2.py $lang eng 1 > ./corrdata/flores_devtest${lang}_srcont_item
exit;
#	/private/home/schwenk/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl < ./data/devtest-flores$lang.eng > ./data/devtest-flores$lang.tok.eng

		python get-bleu.py -r ./data/ref-devtest-flores$lang.eng -b ./data/translation-devtest-flores$lang.$lang-eng.eng > ./corrdata/flores_devtest${lang}_bleu
#		sacrebleu ./data/ref-devtest-flores$lang.eng < ./data/translation-devtest-flores$lang.$lang-eng.eng > ./corrdata/flores_devtestx${lang}_totalbleu

	#
			paste -d' ' ./corrdata/flores_devtest${lang}_srcont ./corrdata/flores_devtest${lang}_bleu > ./corrdata/combinedfloresscores$lang

	#
	sed -i '1s/^/src bleu\n/' ./corrdata/combinedfloresscores$lang

	
	corr=`python correlation.py ./corrdata/combinedfloresscores$lang`

	echo "$lang $corr"

done
