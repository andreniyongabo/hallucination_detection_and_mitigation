#!/bin/bash
src=eng
tgt=zul
pair=$src-$tgt
finalsrc=eng 
finaltgt=zul
finalpair=$finalsrc-$finaltgt
tmpdir=/private/home/angelafan/nllb/fairseq-py/process_holger_bitexts/$pair
finaldir=/checkpoint/angelafan/nllb_data/low_res/noised_bitexts_filtered_share/$finalpair

mkdir -p $tmpdir
mkdir -p $finaldir

echo " - tmp: $tmpdir"

# get texts, train SPM and apply it
ProcTSV () {
    p_inpf=$1
    p_l=$2
    p_pos=$3

    txt=$tmpdir/txt.$p_l
    tok=$tmpdir/tok.$p_l
    spm=$tmpdir/spm.50k.$p_l

    # get text
    echo " - text $p_inpf in language $p_l at $p_pos"
    (zcat /large_experiments/mmt/mining/data/mini-mine1/bitexts/$pair.bitextf.tsv.gz             | head -100000000             | awk -vT=1.06 '{if ($1>=T) print}'             | cut -f$p_pos             | /private/home/schwenk/projects/mlenc/tools-external/moses-tokenizer/tokenizer/remove-non-printing-char.perl | /private/home/schwenk/projects/mlenc/tools-external/moses-tokenizer/tokenizer/normalize-punctuation.perl -l  $p_l | /private/home/schwenk/projects/mlenc/tools-external/moses-tokenizer/tokenizer/deescape-special-chars.perl             > $txt) >> $tmpdir/process.log 2>&1

    # train SPM
    if [ ! -s $spm.model ] ; then
        echo " - train SPM"
        /private/home/angelafan/nllb/fairseq-py/scripts/spm_train.py --input=$txt --model_prefix=$spm --vocab_size=50000 --character_coverage=0.999995 --model_type=unigram --seed_sentencepiece_size=5000000 --shuffle_input_sentence=true --input_sentence_size=5000000 --num_threads=20 >> $tmpdir/process.log 2>&1
    fi

    # apply SPM
    echo " - apply SPM"
    /private/home/angelafan/nllb/fairseq-py/scripts/spm_encode.py --model=$spm.model --inputs=$txt --output_format=piece --outputs=$tok >> $tmpdir/process.log 2>&1
}

ProcTSV /large_experiments/mmt/mining/data/mini-mine1/bitexts/$pair.bitextf.tsv.gz $src 2
ProcTSV /large_experiments/mmt/mining/data/mini-mine1/bitexts/$pair.bitextf.tsv.gz $tgt 3

# filter bitexts on (relative) length
echo " - filter"
perl /private/home/angelafan/mmt/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 2.5 $tmpdir/tok $src $tgt $tmpdir/filter 1 250 >> $tmpdir/process.log 2>&1


# process valid and test, adapted for flores1010 N-way
for fn in "dev" "devtest" ; do
    for ll in $src $tgt ; do
         echo " - apply spm $fn"
         echo /private/home/angelafan/flores101_dataset/$fn/$ll.$fn
         (cat /private/home/angelafan/flores101_dataset/$fn/$ll.$fn | /private/home/schwenk/projects/mlenc/tools-external/moses-tokenizer/tokenizer/remove-non-printing-char.perl | /private/home/schwenk/projects/mlenc/tools-external/moses-tokenizer/tokenizer/normalize-punctuation.perl -l  $ll | /private/home/schwenk/projects/mlenc/tools-external/moses-tokenizer/tokenizer/deescape-special-chars.perl  > $tmpdir/txt.$fn.$ll) >> $tmpdir/process.log 2>&1
         /private/home/angelafan/nllb/fairseq-py/scripts/spm_encode.py --model=$tmpdir/spm.50k.$ll.model --output_format=piece --inputs=$tmpdir/txt.$fn.$ll --outputs=$tmpdir/tok.$fn.$ll
    done
done

# binarize train, dev and test
echo " - binarizing "
rm $finaldir/*dict*
python3 /private/home/angelafan/nllb/fairseq-py/fairseq_cli/preprocess.py --source-lang $finalsrc --target-lang $finaltgt --trainpref $tmpdir/filter --validpref $tmpdir/tok.dev --testpref $tmpdir/tok.devtest --destdir $finaldir --workers 20 > $finaldir/process.log 2>&1

# binarize noised version of data
echo " - binarizing noised version"
rm $finaldir/*dict*
wc -l $tmpdir/noise*
head $tmpdir/noise*
python3 /private/home/angelafan/nllb/fairseq-py/fairseq_cli/preprocess.py --source-lang $finalsrc --target-lang $finaltgt --trainpref $tmpdir/filter_noise --validpref $tmpdir/devnoise --testpref $tmpdir/devtestnoise --destdir $finaldir --workers 20 --nwordssrc 50000 --nwordstgt 50000 > $finaldir/process.log 2>&1
