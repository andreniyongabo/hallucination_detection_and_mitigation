gpu="${GPU:-2}"
export S3_DUMMY_DIR="s3://fairusersglobal/users/sshleifer/dummy_moe_v2"
fs3cmd rm --recursive $S3_DUMMY_DIR
rm -rf dummy_moe_v2
mkdir -p dummy_moe_v2
source scripts/moe_cmd.sh
moe_cmd $gpu --moe-expert-count 4 --moe-freq 1 --restore-file x.pt  --save-dir dummy_moe_v2 --s3-dir $S3_DUMMY_DIR --save-async $@

