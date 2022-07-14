#gpu="${GPU:-2}"
source scripts/moe_cmd.sh
moe_cmd 4 --moe-expert-count 4 --moe-freq 1 --restore-file x.pt --symlink --save-dir moe_symlink $@
#base_cmd 2 --moe-expert-count 4 --moe-freq 2
ls -halt moe_symlink
