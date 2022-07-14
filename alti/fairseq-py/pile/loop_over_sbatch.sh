MODEL=/large_experiments/moe/shru/moe_lm/top2_256e_sv/top2_256e_sv.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu72000.s1.ngpu64/checkpoint_last.pt
ls /private/home/myleott/data/data-bin/ThePile | while read SETPATH; do
  SETNAME=$(echo $SETPATH | tr -d '/');
  echo $SETNAME;
  if [ $SETNAME == "Ubuntu_IRC" ] || [ $SETNAME == "PhilPapers" ] || [ $SETNAME == "NIH_ExPorter" ] || [ $SETNAME == "HackerNews" ] || \
  [ $SETNAME == "EuroParl" ] || [ $SETNAME == "Enron_Emails" ] || [ $SETNAME == "BookCorpus" ] ; then
  echo "skipped"
  else
  sbatch --job-name=$NETNAME --gpus-per-node=8 --nodes=1 --ntasks-per-node=1 --cpus-per-task=80 --mem 470G --time=1440 \
    --constraint volta32gb --partition moe,dev,learnfair --wrap "bash scripts/moe_lm/thepile_eval/eval_ppl.sh $SETNAME $MODEL" ; \
  fi
done
