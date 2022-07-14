```bash
grep -A1 . /scratch/rc_train_big.bpe | grep -v "^--$" > /scratch/rc.filtered.train.bpe
```

```bash
python CC_NEWS/shard_docs.py /scratch/rc.filtered.train.bpe --num-shards 40 --save-dir /scratch/rc_40_shard
mkdir -p /scratch/rc-bin-40

export DICT=/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/dict.txt
export LASTSHARD=39
# preprocess with mmap dataset
for SHARD in $(seq 0 $LASTSHARD);
  do fairseq-preprocess --only-source \
  --dataset-impl mmap \
  --trainpref /scratch/rc_40_shard/shard$SHARD/train.txt \
  --destdir /scratch/rc-bin-40/shard$SHARD --workers 60 \
  --srcdict $DICT
  cp $DICT /scratch/rc-bin-40/shard$SHARD/dict.txt
  done
```

### Copy Valid files
```bash
export LASTSHARD=39
# preprocess with mmap dataset
for SHARD in $(seq 0 $LASTSHARD);
  do 
    cp rc-bin/shard0/valid* rc-bin-40/shard$SHARD/;
  done

```


### Train
```bash
DATA_BIN=/private/home/sshleifer/rc-bin-40
NUM_DATA_SHARDS=40
DATA_DIR="${DATA_BIN}/shard0"
for i in $(seq 1 $(($NUM_DATA_SHARDS-1)));
  do
    DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
  done
bash scripts/moe_lm/train_64experts_32gpus.sh $DATA_DIR 40_shard


expand_data_dir () {
    DATA_BIN=$1
    NUM_DATA_SHARDS=$2
    DATA_DIR="${DATA_BIN}/shard0"
    for i in $(seq 1 $(($NUM_DATA_SHARDS-1)));
      do
        DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
      done
    echo $DATA_DIR
}
```



