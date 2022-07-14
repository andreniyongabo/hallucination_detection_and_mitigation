#!/bin/bash
#
BDIR="/large_experiments/nllb/mmt/data/monolingual"
# SOURCE="$BDIR/source"
SOURCE="/private/home/angelafan/nllb/data_work/fairseq-py/examples/nllb/mining/monolingual/source"
RAW="$BDIR/raw"
SPLIT="$BDIR/proc-split-meta-test"
FILTER="$BDIR/proc-filtered-meta-test"
SCRIPT="$BDIR/proc-script-meta-test"
LID="$BDIR/proc-lid-meta-test"
DEDUP="$BDIR/proc-dedup-meta-test"

# filter params
MIN_SCHAR=10     # minimal number of characters for each sentence
MAX_SCHAR=500    # maximal number of characters for each sentence
MAX_PUNCT=0.2    # max fraction of punctuations in each sentence
MAX_NUMBER=0.2   # max fraction of numbers in each sentence
META_FIELDS=2   # number of tab separated fields with meta information (corpus + line Nb)

MOSES_TOOLS="$SOURCE/moses-tools"
NORM_PUNC="$MOSES_TOOLS/normalize-punctuation.perl"
REM_NON_PRINT_CHAR="$MOSES_TOOLS/remove-non-printing-char.perl"

# which LID model to use
LID_MODEL_DATE="2022-02-18"

# which file lists the languages we are going to run this on?
LIST_OF_LANGS="langs_test.txt"

#---------------------------------
function MKDIR () {
    dir=$1
    if [ ! -d $dir ] ; then
        mkdir -p $dir
    fi
}

#---------------------------------
function SentSplit () {
    lang=$1
    inpf=$2
    outf=$3
    logf=$4
    corpus=`basename $inpf | sed -e 's/\..*//'`
    if [ -s $outf ] ; then
        echo " - segment $outf EXISTS" > /dev/null
    else
        echo " - segment $inpf"
        # we add iconv to remove illegal UTF-8 characaters
        script=${outf/.xz/.sh}
        cat << EOF > $script
#!/bin/bash
        (xzcat $inpf \
            | iconv -f UTF-8 -t UTF-8 -c \
            | python3 $SOURCE/SentenceSplit.py --lang $lang --split-algo default --add-meta $corpus \
            | xz --compress --stdout > $outf ) > $logf 2>&1
        /bin/rm $script
EOF
        chmod 755 $script
        if $SUBMIT ; then
            name="split.`basename ${outf/.xz/}`"
            sbatch -J $name --partition=$QUEUE  \
                --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=8 \
                --mem=50G --time=1200 $script
        else
            $script &
        fi
    fi
}

#---------------------------------
function NormalizeFilter () {
    lang=$1
    inpf=$2
    outf=$3
    logf=$4
    if [ ! -s $inpf ] ; then
        echo " - filter input $inpt missing"
        return
    fi

    if [ -s $outf ] ; then
        echo " - filter $outf EXISTS" > /dev/null
    else
        echo " - filter $inpf"
        # map ISO3 languages to ISO2
        # punctuation normalizer has rules for English and (fr,es,de,cz)
        iso2="en"
        script=${outf/.xz/.sh}
        cat << EOF > $script
#!/bin/bash
        tmpf=/tmp/filter.\$\$
        (xzcat $inpf | cut -f1,2 > \$tmpf.1; \
         xzcat $inpf | cut -f3- \
            | $NORM_PUNC -l $iso2 \
            | $REM_NON_PRINT_CHAR \
            | python3 $SOURCE/FilterText.py --lang $lang --verbose \
                --min-chars $MIN_SCHAR --max-chars $MAX_SCHAR \
                --max-punct-ratio $MAX_PUNCT \
                --max-number-ratio $MAX_NUMBER \
                --meta-fields 0 > \$tmpf.2; \
            paste \$tmpf.1 \$tmpf.2 | xz --compress --stdout > $outf ) > $logf 2>&1
        /bin/rm \$tmpf.1 \$tmpf.2
        /bin/rm $script
EOF
        chmod 755 $script
        if $SUBMIT ; then
            name="filt.`basename ${outf/.xz/}`"
            sbatch -J $name --partition=$QUEUE  \
                --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=8 \
                --mem=50G --time=1200 $script
        else
            $script &
        fi
    fi
}

#---------------------------------
function LID () {
    lang=$1
    inpf=$2
    outf=$3
    logf=$4
    P=0.01

    if [ ! -s $inpf ] ; then
        echo " - LID input $inpt missing"
        return
    fi

    if [ -s $outf ] ; then
        echo " - LID $outf EXISTS" > /dev/null
    else
        echo " - LID on $inpf"
        script=${outf/.xz/.sh}
        cat << EOF > $script
#!/bin/bash
        (xzcat $inpf \
            | python3 $SOURCE/predict_nllb_lid.py \
                --model-date $LID_MODEL_DATE --filter-mode \
                --meta-fields $META_FIELDS \
            | sed -e 's/^__label__//' \
            | awk -vL=$lang -vP=$P -F'\t' '{ninp++; if (\$1==L && \$2>=P) {printf("%s\t%s\t%s\n",\$3,\$4,\$5); nout++} } \
                                            END {printf(" - %d sentences, %d (%5.2f%%) have correct %s language\n", \
                                                 ninp, nout, 100.0*nout/ninp, L) > "/dev/stderr"} ' \
            | xz --compress --stdout > $outf ) > $logf 2>&1
        #/bin/rm $script
EOF
        chmod 755 $script
        if $SUBMIT ; then
            name="lid.`basename ${outf/.xz/}`"
            sbatch -J $name --partition=$QUEUE  \
                --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=8 \
                --mem=50G --time=1200 $script
        else
            $script &
        fi
    fi
}

#---------------------------------
function FilterScript () {
    lang=$1
    inpf=$2
    outf=$3
    logf=$4
    S="`grep \"^$lang\" $SOURCE/language_scripts_200.tsv | cut -f2`"

    if [ ! -s $inpf ] ; then
        echo " - FilterScript input $inpt missing"
        return
    fi

    if [ -s $outf ] ; then
        echo " - FilterScript $outf EXISTS" > /dev/null
    else
        echo " - FilterScript on $inpf"
        script=${outf/.xz/.sh}
        cat << EOF > $script
#!/bin/bash
        (xzcat $inpf \
            | python3 $SOURCE/predict_script.py --filter-mode --meta-fields $META_FIELDS \
            | awk -vS=$S -F'\t' '{ ninp++; if (\$1==S) {printf("%s\t%s\t%s\n",\$2,\$3,\$4); nout++} } \
                                   END {printf(" - %d sentences, %d (%5.2f%%) have correct %s script\n", \
                                               ninp, nout, 100.0*nout/ninp, S) > "/dev/stderr"} ' \
            | xz --compress --stdout > $outf ) > $logf 2>&1
        #/bin/rm $script
EOF
        chmod 755 $script
        if $SUBMIT ; then
            name="script.`basename ${outf/.xz/}`"
            sbatch -J $name --partition=$QUEUE  \
                --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=4 \
                --mem=50G --time=1200 $script
        else
            $script &
        fi
    fi
}

#---------------------------------
function Deduplicate () {
    lang=$1
    inp_dir=$2
    out_dir=$3
    outf=$out_dir/$lang.xz
    metaf=$out_dir/$lang.meta.xz
    logf=$out_dir/$lang.log
    if [ -s $outf ] ; then
        echo " - dedup $outf EXISTS" > /dev/null
    else
        echo " - dedup $inp_dir"
        script=${outf/.xz/.sh}
        cat << EOF > $script
#!/bin/bash
        (xzcat $inp_dir/*.$lang.xz \
          | python3 $SOURCE/Dedup.py --verbose \
                --meta-fields $META_FIELDS --meta-out $metaf \
          | xz --compress --stdout > $outf ) > $logf 2>&1
        #/bin/rm $script
EOF
        chmod 755 $script
        if $SUBMIT ; then
            name="dedup.`basename ${outf/.xz/}`"
            sbatch -J $name --partition=$QUEUE  \
                --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=4 \
                --mem=50G --time=1200 $script
        else
            $script &
        fi
    fi
}

#----------------------

QUEUE="devaccel"
QUEUE="learnfair"
SUBMIT=false

# optionally handle a specifi corpus only:
#C="paracrawl9"
#C="paracrawl20211101"
C=""

langs=()

for l in `cat $LIST_OF_LANGS` #${langs[@]}
do
    MKDIR $SPLIT/$l
    MKDIR $FILTER/$l
    MKDIR $SCRIPT/$l
    MKDIR $LID/$l
    MKDIR $DEDUP/$l

    if [ ! -d $RAW/$l ] ; then
        echo " - no data for $l"
        continue
    fi

    for c in $RAW/$l/${C}*.$l.xz ; do
        fn=`basename $c`
        SentSplit $l $c $SPLIT/$l/$fn $SPLIT/$l/${fn/.xz/.log}
    done
    if ! $SUBMIT; then
        wait # wait for all parallel jobs to finish
    fi

    for c in $SPLIT/$l/${C}*.$l.xz ; do
        fn=$FILTER/$l/`basename $c`
        NormalizeFilter $l $c $fn ${fn/.xz/.log}
    done
    if ! $SUBMIT; then
        wait # wait for all parallel jobs to finish
    fi

    for c in $FILTER/$l/${C}*.$l.xz ; do
        fn=$SCRIPT/$l/`basename $c`
        FilterScript $l $c $fn ${fn/.xz/.log}
    done
    if ! $SUBMIT; then
        wait # wait for all parallel jobs to finish
    fi

    for c in $SCRIPT/$l/${C}*.$l.xz ; do
        fn=$LID/$l/`basename $c`
        LID $l $c $fn ${fn/.xz/.log}
    done
    if ! $SUBMIT; then
        wait # wait for all parallel jobs to finish
    fi

    Deduplicate $l $LID/$l $DEDUP/$l

done
