for fil in news.2020.en.shuffled.deduped; do
  python examples/wmt21/backtranslation/run_backtranslation_slurm.py \
    --paths /private/home/angelafan/wmt21/wmt20/monolingual/collated/en_XX/${fil} \
    --direction en-ha \
    --mode preprocess
done
