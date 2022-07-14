gpu=${1-8}
#export S3_DUMMY_DIR="s3://fairusersglobal/users/sshleifer/dummy_moe_v2"
#fs3cmd rm --recursive $S3_DUMMY_DIR
rm -rf dummy_moe_local
mkdir -p dummy_moe_local
fairseq-train \
  --distributed-world-size $gpu \
  --save-dir dummy_moe_local \
  --distributed-port 14666 /private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin \
  --train-subset train11 \
  --memory-efficient-fp16 \
  --num-workers 1 \
  --ddp-backend no_c10d \
  --validate-interval-updates 100 \
  --save-interval-updates 25 \
  --total-num-update 200 \
  --max-update 200 \
  --warmup-updates 2000 \
  --no-epoch-checkpoints \
  --keep-best-checkpoints 1 \
  --keep-interval-updates 1 \
  --task language_modeling \
  --sample-break-mode none \
  --tokens-per-sample 1024 \
  --arch transformer_lm_gpt2_tiny \
  --decoder-layers 4 \
  --criterion moe_cross_entropy \
  --moe-gate-loss-wt 0.01 \
  --moe-gate-loss-combine-method sum \
  --moe-second-expert-policy all \
  --moe-gating-use-fp32 \
  --moe-freq 1 \
  --moe-expert-count $gpu \
  --share-decoder-input-output-embed \
  --optimizer adam \
  --adam-eps 1e-06 \
  --clip-norm 0.1 \
  --lr-scheduler polynomial_decay \
  --lr 0.002 \
  --max-sentences 2 \
  --max-sentences-valid 2 \
  --pad-to-fixed-length \
  --required-batch-size-multiple 1 \
  --update-freq 8 \
  --log-format json \
  --log-interval 50
