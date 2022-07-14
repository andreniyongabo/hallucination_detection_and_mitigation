#!/bin/bash


EXPERIMENT_NAME="2021-09-28-13-50-push-fasttext-2"

# temporary new expriment root
# EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"
EXPERIMENT_FOLDER="/checkpoint/celebio/projects/nlp/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER
FT_LOGS="$RESULT_FOLDER/ft_logs"
mkdir -p $FT_LOGS


PREV_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile"
TRAIN_DATA="$PREV_TRAIN_FOLDER/data/train/all.txt"
VALID_DATA="$PREV_TRAIN_FOLDER/data/valid/all.txt"




FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"

train_fasttext_cluster() {

    sbatch --job-name lid_11 \
        --partition nllb --comment lid_11 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_11.log \
        --error $FT_LOGS/train_11.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.11 -lr 0.1 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_12 \
        --partition nllb --comment lid_12 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_12.log \
        --error $FT_LOGS/train_12.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.12 -lr 0.2 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_13 \
        --partition nllb --comment lid_13 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_13.log \
        --error $FT_LOGS/train_13.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.13 -lr 0.3 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_15 \
        --partition nllb --comment lid_15 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_15.log \
        --error $FT_LOGS/train_15.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.15 -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_18 \
        --partition nllb --comment lid_18 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_18.log \
        --error $FT_LOGS/train_18.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.18 -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_20 \
        --partition nllb --comment lid_20 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_20.log \
        --error $FT_LOGS/train_20.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.20 -lr 1.0 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_21 \
        --partition nllb --comment lid_21 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_21.log \
        --error $FT_LOGS/train_21.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.21 -lr 1.5 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_22 \
        --partition nllb --comment lid_22 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_22.log \
        --error $FT_LOGS/train_22.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.22 -lr 2.0 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 256 -loss ova -bucket 10000000 -thread 10"




    sbatch --job-name lid_31 \
        --partition nllb --comment lid_31 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_31.log \
        --error $FT_LOGS/train_31.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.31 -lr 0.1 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_32 \
        --partition nllb --comment lid_32 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_32.log \
        --error $FT_LOGS/train_32.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.32 -lr 0.2 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_35 \
        --partition nllb --comment lid_35 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_35.log \
        --error $FT_LOGS/train_35.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.35 -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_38 \
        --partition nllb --comment lid_38 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_38.log \
        --error $FT_LOGS/train_38.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.38 -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss ova -bucket 10000000 -thread 10"


    sbatch --job-name lid_41 \
        --partition nllb --comment lid_41 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_41.log \
        --error $FT_LOGS/train_41.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.41 -lr 0.1 -minn 2 -maxn 5 -minCount 1000 -epoch 3 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_42 \
        --partition nllb --comment lid_42 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_42.log \
        --error $FT_LOGS/train_42.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.42 -lr 0.2 -minn 2 -maxn 5 -minCount 1000 -epoch 3 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_45 \
        --partition nllb --comment lid_45 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_45.log \
        --error $FT_LOGS/train_45.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.45 -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 3 -dim 256 -loss ova -bucket 10000000 -thread 10"

    sbatch --job-name lid_48 \
        --partition nllb --comment lid_48 \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 \
        --output $FT_LOGS/train_48.log \
        --error $FT_LOGS/train_48.err \
        --mem=100G \
        --open-mode append --no-requeue --wrap \
        "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DATA -output $RESULT_FOLDER/model.48 -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 3 -dim 256 -loss ova -bucket 10000000 -thread 10"

}


train_fasttext_cluster
