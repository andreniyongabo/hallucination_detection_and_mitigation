#!/bin/bash
#
# Evaluate all checkpoints for a language pair on de/v/test used during binarization
# (usually Flroes-101)
# Keeps track which checkpoint had already been evaluated and stores result files
# with BLEU scores for each checkpoint.
# Result parsing can be done with the tool "ana.sh"
#
# NLLB, Holger Schwenk
#
# usage:  eval.sh LANGUAGE-PAIR
#         e.g. eval.sh hin-eng
#
# Directory structure with respect to $bdir
#  bitexts/     bitexts, e.g. eng-hau.bitextf.tsv.gz (alphabetical order)
#  bin/         binarized bitexts and FLORES  dev/test sets
#               (see script preproc-binarize-mined.sh)
#  models/      trained FAIRSEQ models


bdir="/private/home/pkoehn/experiment/mining"
FSEQ="$HOME/project/fairseq-internal"

QUEUE="learnfair"
COMMENT=""

####################################################################################
# Eval already binarized data, i.e. ted valid or test
# 
EvalBin () {
    l1=$1
    l2=$2
    part=$3
    ID=$4
    max_epoch=$5
    corpus="flores101"  # that's the one we have binarized as valid/test

    datadir="$bdir/$ID/bin"
    mdir="$bdir/$ID/model.$l1-$l2.epoch$max_epoch"

    N=0
    if [ -d $mdir ] ; then
        N=`ls $mdir/ | grep -c 'checkpoint[0-9]*.pt'`
        if [ $N -gt 0 ] ; then
            N=`ls $mdir/*pt | sed -e 's/.*checkpoint//' -e 's/\..*//' | sort -n | tail -1`
        fi
    fi
    if [ $N == 0 ] ; then
        echo " - no checkpoints found in $mdir"
        return
    fi
    name="$corpus.$part"
    if [ `ls $mdir | grep -c $name.bleu` -gt 0 ] ; then
        Nlow=`ls $mdir/*$name.bleu | sed -e 's/.*checkpoint//' -e 's/\..*//' | sort -n | tail -1`
        if [ -z $Nlow ] ; then Nlow=1; else let Nlow=Nlow+1; fi
    else
        Nlow=1
    fi
    echo "EVAL $corpus.$part $l1-$l2 on $Nlow-$N x $mdir"

    if [ $Nlow -gt $N ] ; then return; fi

    script="$mdir/eval.$name.sh"
    cat << EOF > $script
#!/bin/bash
. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate fairseq-20210318

bdir="/private/home/pkoehn/experiment/mining/$ID"
FSEQ="$HOME/project/fairseq-internal"

i=\$SLURM_ARRAY_TASK_ID
chkpt="$mdir/checkpoint\$i.pt"
out=\${chkpt//.pt/.$name.out}
bleu=\${chkpt//.pt/.$name.bleu}

if [ ! -s \$out ] ; then
    python3 $FSEQ/fairseq_cli/generate.py \
          $datadir --gen-subset $part \
          --source-lang $l1 --target-lang $l2 \
          --no-progress-bar --path \$chkpt \
          --batch-size 128 --beam 5 \
          > \$out
fi

if [ -s \$out -a ! -s \$bleu ] ; then
    grep '^T' \$out | cut -f2 | sed -e 's/ //g' -e 's/▁/ /g' > \$out.ref
    grep '^H' \$out | cut -f3 | sed -e 's/ //g' -e 's/▁/ /g' > \$out.hyp
    /private/home/pkoehn/mosesdecoder/scripts/generic/multi-bleu-detok.perl \$out.ref < \$out.hyp > \$bleu
    /bin/rm \$out.{hyp,ref}
fi
EOF

    chmod 755 $script
    if $SUBMIT ; then
        sbatch -J "beval.$part.$l1-$l2" \
            --partition=$QUEUE --comment "$COMMENT" \
            --nodes=1 --gpus-per-node=1 --cpus-per-task=2 \
            --ntasks-per-node=1 --time=30 \
            --array $Nlow-$N $script
    else
        echo "TODO"
    fi
}

EvalBinLocal () {
    l1=$1
    l2=$2
    part=$3
    ID=$4
    corpus="flores101"  # that's the one we have binarized as valid/test

    datadir="$bdir/$ID/bin"
    mdir="$bdir/$ID/model"

    N=0
    if [ -d $mdir ] ; then
        N=`ls $mdir/ | grep -c 'checkpoint[0-9]*.pt'`
        if [ $N -gt 0 ] ; then
            N=`ls $mdir/*pt | sed -e 's/.*checkpoint//' -e 's/\..*//' | sort -n | tail -1`
        fi
    fi
    if [ $N == 0 ] ; then
        echo " - no checkpoints found in $mdir"
        return
    fi
    name="$corpus.$part"
    if [ `ls $mdir | grep -c $name.bleu` -gt 0 ] ; then
        Nlow=`ls $mdir/*$name.bleu | sed -e 's/.*checkpoint//' -e 's/\..*//' | sort -n | tail -1`
        if [ -z $Nlow ] ; then Nlow=1; else let Nlow=Nlow+1; fi
    else
        Nlow=1
    fi

    N=100
    Nlow=1
    # last to first
    for i in $(seq $N -1 $Nlow) ; do
        echo "EVAL local $corpus.$part $l1-$l2 on epoch $i"
        chkpt="$mdir/checkpoint$i.pt"
        out=${chkpt//.pt/.$name.out}
        bleu=${chkpt//.pt/.$name.bleu}

        if [ ! -s $out ] ; then
            python3 $FSEQ/fairseq_cli/generate.py \
                  $datadir --gen-subset $part \
                  --source-lang $l1 --target-lang $l2 \
                  --no-progress-bar --path $chkpt \
                  --batch-size 128 --beam 5 \
                  > $out 2> /dev/null
        fi

        if [ -s $out -a ! -s $bleu ] ; then
            grep '^T' $out | cut -f2 | sed -e 's/ //g' -e 's/▁/ /g' > $out.ref
            grep '^H' $out | cut -f3 | sed -e 's/ //g' -e 's/▁/ /g' > $out.hyp
            $HOME/bin/multi-bleu-detok.perl $out.ref < $out.hyp > $bleu
            /bin/rm $out.{hyp,ref}
        fi
    done
}


####################################################################################

# parameters: l1 / l2 / id
SUBMIT=true
EvalBin $1 $2 "valid" $3 $4
EvalBin $1 $2 "test" $3 $4
