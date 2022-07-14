#!/bin/bash

BDIR="/large_experiments/nllb/mmt/data/monolingual"
SOURCE="$BDIR/source"
RAW="$BDIR/raw"
SPLIT="$BDIR/proc-split-meta"
FILTER="$BDIR/proc-filtered-meta"
SCRIPT="$BDIR/proc-script-meta"
LID="$BDIR/proc-lid-meta"
DEDUP="$BDIR/proc-dedup-meta"

# filter params
MIN_SCHAR=10     # minimal number of characters for each sentence
MAX_SCHAR=500    # maximal number of characters for each sentence
MAX_PUNCT=0.2    # max fraction of punctuations in each sentence
MAX_NUMBER=0.2   # max fraction of numbers in each sentence
META_FIELDS=2   # number of tab separated fields with meta information (corpus + line Nb)

MOSES_TOOLS="$SOURCE/moses-tools"
NORM_PUNC="$MOSES_TOOLS/normalize-punctuation.perl"
REM_NON_PRINT_CHAR="$MOSES_TOOLS/remove-non-printing-char.perl"

#---------------------------------
function MKDIR () {
    dir=$1
    if [ ! -d $dir ] ; then
        mkdir -p $dir
    fi
}

#---------------------------------
function SetBeginEnd () {
    idir=$1
    odir=$2
    C=$3
    # find first input file for which output is missing
    for idx_beg in $(seq 0 999); do
        fn=`printf $C $idx_beg`
        if [ -s $idir/$fn -a ! -s $odir/$fn ] ; then
            break
        fi
    done
    idx_end=$idx_beg
    if [ $idx_beg -lt 999 ] ; then
        # find last existing input file
        for idx_end in $(seq 999 -1 0); do
            fn=`printf $C $idx_end`
            if [ -s $idir/$fn ] ; then
                break
            fi
        done
    fi
}

#---------------------------------
function SentSplitArray () {
    lang=$1
    corpus="cc200xl_v1"
    SetBeginEnd $RAW/$lang $SPLIT/$lang "${corpus}_p%03d.$lang.xz"
    if [ $idx_beg -lt 999 ] ; then
        echo " - split $corpus in $lang  $idx_beg-$idx_end"
    else
        echo " - split $corpus in $lang  all done"
        return
    fi

    script="$SPLIT/$lang/$corpus.sh"
    cat << EOF > $script
#!/bin/bash
    id=\`printf "%03d" \$SLURM_ARRAY_TASK_ID\`
    inpf="$RAW/$lang/${corpus}_p\$id.$lang.xz"
    outf="$SPLIT/$lang/${corpus}_p\$id.$lang.xz"
    logf=\${outf/.xz/.log}
    if [ -s \$inpf -a ! -s \$outf ] ; then
        (xzcat \$inpf \
            | iconv -f UTF-8 -t UTF-8 -c \
            | python3 $SOURCE/SentenceSplit.py --lang $lang --split-algo default --add-meta $corpus \
            | xz --compress --stdout > \$outf ) > \$logf 2>&1
    fi
EOF
    chmod 755 $script
    name="split.$corpus"
    sbatch -J $name --partition=$QUEUE --array="$idx_beg-$idx_end" \
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=8 --time=600 $script
}

#---------------------------------
function NormalizeFilterArray () {
    lang=$1
    corpus="cc200xl_v1"
    SetBeginEnd $SPLIT/$lang $FILTER/$lang "${corpus}_p%03d.$lang.xz"
    if [ $idx_beg -lt 999 ] ; then
        echo " - filter $corpus in $lang  $idx_beg-$idx_end"
    else
        echo " - filter $corpus in $lang  all done"
        return
    fi

    # map ISO3 languages to ISO2
    # punctuation normalizer has rules for English and (fr,es,de,cz)
    iso2="en"
    script="$FILTER/$lang/$corpus.sh"
    cat << EOF > $script
#!/bin/bash
    id=\`printf "%03d" \$SLURM_ARRAY_TASK_ID\`
    inpf="$SPLIT/$lang/${corpus}_p\$id.$lang.xz"
    outf="$FILTER/$lang/${corpus}_p\$id.$lang.xz"
    logf=\${outf/.xz/.log}
    if [ -s \$inpf -a ! -s \$outf ] ; then
        tmpf=/tmp/filter.\$\$
        (xzcat \$inpf | cut -f1,2 > \$tmpf.1; \
         xzcat \$inpf | cut -f3- \
            | $NORM_PUNC -l $iso2 \
            | $REM_NON_PRINT_CHAR \
            | python3 $SOURCE/FilterText.py --lang $lang --verbose \
                --min-chars $MIN_SCHAR --max-chars $MAX_SCHAR \
                --max-punct-ratio $MAX_PUNCT \
                --max-number-ratio $MAX_NUMBER \
                --meta-fields 0 > \$tmpf.2; \
            paste \$tmpf.1 \$tmpf.2 | xz --compress --stdout > \$outf ) > \$logf 2>&1
        /bin/rm \$tmpf.1 \$tmpf.2
    fi
EOF
    chmod 755 $script
    name="filter.$corpus"
    sbatch -J $name --partition=$QUEUE  --array="$idx_beg-$idx_end" \
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=8 \
        --mem=50G --time=600 $script
}


#---------------------------------
function FilterScriptArray () {
    lang=$1
    corpus="cc200xl_v1"
    SetBeginEnd $FILTER/$lang $SCRIPT/$lang "${corpus}_p%03d.$lang.xz"
    if [ $idx_beg -lt 999 ] ; then
        echo " - script $corpus in $lang  $idx_beg-$idx_end"
    else
        echo " - script $corpus in $lang  all done"
        return
    fi

    S="`grep \"^$lang\" $SOURCE/language_scripts.tsv | cut -f2`"

    script="$SCRIPT/$lang/$corpus.sh"
    cat << EOF > $script
#!/bin/bash
    id=\`printf "%03d" \$SLURM_ARRAY_TASK_ID\`
    inpf="$FILTER/$lang/${corpus}_p\$id.$lang.xz"
    outf="$SCRIPT/$lang/${corpus}_p\$id.$lang.xz"
    logf=\${outf/.xz/.log}
    if [ -s \$inpf -a ! -s \$outf ] ; then
        (xzcat \$inpf \
            | python3 $SOURCE/predict_script.py --filter-mode --meta-fields $META_FIELDS \
            | awk -vS=$S -F'\t' '{ ninp++; if (\$1==S) {printf("%s\t%s\t%s\n",\$2,\$3,\$4); nout++} } \
                                   END {printf(" - %d sentences, %d (%5.2f%%) have correct %s script\n", \
                                               ninp, nout, 100.0*nout/ninp, S) > "/dev/stderr"} ' \
            | xz --compress --stdout > \$outf ) > \$logf 2>&1
    fi
EOF
    chmod 755 $script
    name="script.$corpus"
    sbatch -J $name --partition=$QUEUE --array="$idx_beg-$idx_end" \
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=4 \
        --mem=50G --time=600 $script
}

#---------------------------------
function LIDArray () {
    lang=$1
    corpus="cc200xl_v1"
    SetBeginEnd $SCRIPT/$lang $LID/$lang "${corpus}_p%03d.$lang.xz"
    if [ $idx_beg -lt 999 ] ; then
        echo " - lid $corpus in $lang  $idx_beg-$idx_end"
    else
        echo " - lid $corpus in $lang  all done"
        return
    fi

    P=0.01
    script="$LID/$lang/$corpus.sh"
    cat << EOF > $script
#!/bin/bash
    id=\`printf "%03d" \$SLURM_ARRAY_TASK_ID\`
    inpf="$SCRIPT/$lang/${corpus}_p\$id.$lang.xz"
    outf="$LID/$lang/${corpus}_p\$id.$lang.xz"
    logf=\${outf/.xz/.log}
    if [ -s \$inpf -a ! -s \$outf ] ; then
        (xzcat \$inpf \
            | python3 $SOURCE/predict_nllb_lid.py \
                --model-date last --filter-mode \
                --meta-fields $META_FIELDS \
            | sed -e 's/^__label__//' \
            | awk -vL=$lang -vP=$P -F'\t' '{ninp++; if (\$1==L && \$2>=P) {printf("%s\t%s\t%s\n",\$3,\$4,\$5); nout++} } \
                                   END {printf(" - %d sentences, %d (%5.2f%%) have correct %s language\n", \
                                               ninp, nout, 100.0*nout/ninp, L) > "/dev/stderr"} ' \
            | xz --compress --stdout > \$outf ) > \$logf 2>&1
    fi
EOF
    chmod 755 $script
    name="lid.$corpus"
    sbatch -J $name --partition=$QUEUE --array="$idx_beg-$idx_end" \
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=0 --cpus-per-task=8 \
        --mem=50G --time=600 $script
}

#---------------------------------
function Deduplicate () {
    lang=$1
    inp_dir=$2
    out_dir=$3
    outf=$out_dir/$lang.xz
    meta=$out_dir/$lang.meta.xz
    logf=$out_dir/$lang.log

    echo "use different implementation to handle large corpora"; exit

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

QUEUE="learnfair"

for l in eng
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

    SentSplitArray $l
    NormalizeFilterArray $l
    FilterScriptArray $l
    LIDArray $l
    #Deduplicate $l $LID/$l $DEDUP/$l

done
