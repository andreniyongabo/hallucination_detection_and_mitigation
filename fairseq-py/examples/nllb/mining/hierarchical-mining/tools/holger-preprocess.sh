#!/bin/bash
#
# Binarize bitexts 
# as well as Flores-101 dev and test

######################################
# Calling parameters
######################################

set -e
lsrc=$1
ltrg=$2
filesrc=$3
filetrg=$4
name=$5
bi_size=${6:-100}       # max size of bitexts
SPM=${7:-50}            # size of SPM vocab in [k]

######################################
# *** START CONFIGURATION HERE ****
######################################

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate fairseq-20210318

bdir="/private/home/pkoehn/experiment/mining/$name-spm$SPM"

# default FLORES-101 test data
tst_dir="/large_experiments/mmt/mining/data/flores101/flores101"

# external tools
FSEQ="$HOME/project/fairseq-internal"
LASER="$HOME/project/laser"

QUEUE="learnfair"; COMMENT="na"

######################################
# *** END CONFIGURATION HERE ****
######################################

# default directory structure
bin_dir="$bdir/bin"  # all binarized data goes here

######################################
# *** Default normalization and binarization settings
######################################

MINE_BASE=""            # naming of potential future versions
MINE_TYPE=""
SPM_SZ=$SPM               # size of SPM vocab in [k]
SPM_TRAIN_SZ=5000000    # size of SPM training data
FILTER_RATIO="2.5"      # bietxts filtering using Moses's clean-corpus-n.perl
FILTER_MIN="1"
FILTER_MAX="250"

# various tools for normalization and processing 
MOSES_BDIR="$HOME/mosesdecoder/scripts"
NORM_PUNC="$MOSES_BDIR/tokenizer/normalize-punctuation.perl -l "
DESCAPE="$MOSES_BDIR/tokenizer/deescape-special-chars.perl"
REM_NON_PRINT_CHAR="$MOSES_BDIR/tokenizer/remove-non-printing-char.perl"
MOSES_FILTER="perl $MOSES_BDIR/training/clean-corpus-n.perl"

for d in $bin_dir ; do
  if [ ! -d $d ] ; then mkdir -p $d ; fi
done

########################################################################
# Main binarization function

Binarize () {
    l1=$1
    l2=$2
    file1=$3
    file2=$4
    SZ=${5:-100}  # default max size
    NWORKERS=20

    if [ ! -f $file1 ] ; then
        echo " - $file1 ERROR no source file found"
        return
    fi
    if [ ! -f $file1 ] ; then
        echo " - $file2 ERROR no target file found"
        return
    fi

    # dir of binarized data
    bi=$l1=$l2
    odir=$bin_dir

    if [ ! -s $odir/train.$bi.$l1.bin ] ; then
      echo "bitexts: `basename $file1` /  `basename $file2`, languages: $l1 - $l2,  max ${SZ}M"
      echo " - binarizing into $odir"
      if [ ! -d $odir ] ; then mkdir -p $odir; fi
      script=$odir/process.sh
      logf=${script//.sh/.log}
      cat > $script << EOF
#!/bin/bash

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate fairseq-20210318

    PATH=/$HOME/tools/sentencepiece/build/src:$PATH
    tmpdir=$bdir/bin/tmp
    mkdir -p \$tmpdir
    echo " - tmp: \$tmpdir"

    # get texts, train SPM and apply it
    Proc () {
        p_inpf=\$1
        p_l=\$2

        txt=\$tmpdir/txt.\$p_l
        tok=\$tmpdir/tok.\$p_l
        spm=$odir/spm.${SPM_SZ}k.\$p_l

        # get text
        echo " - text \$p_inpf in language \$p_l"
        (cat \$p_inpf \
            | head -${SZ}000000 \
            | $REM_NON_PRINT_CHAR | $NORM_PUNC \$p_l | $DESCAPE \
            > \$txt) >> $logf 2>&1

        # train SPM
        if [ ! -s \$spm.model ] ; then
            echo " - train SPM"
            spm_train --input=\$txt \
                --model_prefix=\$spm --vocab_size=${SPM_SZ}000 \
                --character_coverage=0.999995 --model_type=unigram \
                --seed_sentencepiece_size=$SPM_TRAIN_SZ \
                --shuffle_input_sentence=true --input_sentence_size=$SPM_TRAIN_SZ \
                --num_threads=20 >> $logf 2>&1
        fi

        # apply SPM
        echo " - apply SPM"
        (cat \$txt | spm_encode -model=\$spm.model --output_format=piece > \$tok) >> $logf 2>&1
    }

    Proc $filesrc $l1
    Proc $filetrg $l2

    # filter bitexts on (relative) length
    echo " - filter"
    $MOSES_FILTER -ratio $FILTER_RATIO \$tmpdir/tok $l1 $l2 \$tmpdir/filter $FILTER_MIN $FILTER_MAX >> $logf 2>&1

    # process valid and test, adapted for flores101 N-way
    for fn in "dev" "test" ; do
        for ll in $l1 $l2 ; do
             echo " - apply spm \$fn"
             (cat $tst_dir.\$fn.\$ll \
                | $REM_NON_PRINT_CHAR | $NORM_PUNC \$ll | $DESCAPE \
                | spm_encode -model=$odir/spm.${SPM_SZ}k.\$ll.model --output_format=piece > \$tmpdir/tok.\$fn.\$ll) >> $logf 2>&1
        done
    done

    # binarize train, dev and test
    if [ -s \$tmpdir/filter.$l1 -a -s \$tmpdir/filter.$l2 ] ; then
        echo " - binarizing into $odir"
        python3 $FSEQ/fairseq_cli/preprocess.py \
                --source-lang $l1 --target-lang $l2 \
                --trainpref \$tmpdir/filter --validpref \$tmpdir/tok.dev --testpref \$tmpdir/tok.test \
                --destdir $odir --workers $NWORKERS >> $logf 2>&1
                #--srcdict $odir/spm.${SPM_SZ}k.$l1.cvocab --tgtdict $odir/spm.${SPM_SZ}k.$l2.cvocab 
    fi
    /bin/rm -rf \$tmpdir
EOF
      chmod 755 $script
      sbatch -J "pre.$bi" --partition="$QUEUE" --comment="$COMMENT" \
           --nodes=1 --ntasks-per-node=1 --gres=gpu:0 --cpus-per-task=$NWORKERS \
           --time=600 ${script}
    fi
}


############################################################################################

Binarize $lsrc $ltrg $filesrc $filetrg $bi_size
