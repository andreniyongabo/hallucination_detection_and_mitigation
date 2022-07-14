MODEL_FODLER=${1}
SRC=${2}
TGT=${3}

cat $MODEL_FODLER/gen_best.out | grep -P "^T-" | cut -f2  > $MODEL_FODLER/gen_best.ref
cat $MODEL_FODLER/gen_best.out | grep -P "^D-" | cut -f3  > $MODEL_FODLER/gen_best.hyp
sacrebleu -l ${SRC}-${TGT} $MODEL_FODLER/gen_best.ref < $MODEL_FODLER/gen_best.hyp