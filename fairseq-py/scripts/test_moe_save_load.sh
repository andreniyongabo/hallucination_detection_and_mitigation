#gpu="${GPU:-2}"
source scripts/moe_cmd.sh

moe_cmd 4 --moe-expert-count 4 --moe-freq 1 \
  --restore-file x.pt \
  --ddp-backend fully_sharded --save-dir dummy_moe_fsdp_v2 \
  --log-interval 5 --max-update 50 \
  --keep-interval-updates 2 \
  "$@" | tee moe_from_scratch.log

moe_cmd 4 --moe-expert-count 4 \
  --moe-freq 1 \
  --ddp-backend fully_sharded --save-dir dummy_moe_fsdp_cont \
  --restore-file dummy_moe_fsdp_v2/checkpoint_1_25.pt \
  --max-update 50 --keep-interval-updates 2 --log-interval 5 | tee moe_cont.log


echo "\n\n\n LOGS"
grep train_inner  from_scratch.log | tail -n 1

echo "\nContinued:"
grep train_inner agbm_cont.log | tail -n 1
