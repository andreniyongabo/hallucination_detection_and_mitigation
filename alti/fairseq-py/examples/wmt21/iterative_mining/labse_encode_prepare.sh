lang=ha
mkdir -p /large_experiments/mmt/wmt21/labse_embeddings/monolingual/${lang}
for corpus in cc100_xl_wmt.lid.txt  cc_wmt.lid.txt  news.2008.ha.filtered.lid.txt  news.2009.ha.filtered.lid.txt  news.2010.ha.filtered.lid.txt  news.2011.ha.filtered.lid.txt  news.2012.ha.filtered.lid.txt  news.2013.ha.filtered.lid.txt  news.2014.ha.filtered.lid.txt  news.2018.ha.filtered.lid.txt  news.2019.ha.filtered.lid.txt  news.2020.M6.ha.filtered.lid.txt  news.2020.ha.shuffled.deduped.lid.txt; do 
  mkdir -p /large_experiments/mmt/wmt21/labse_embeddings/monolingual/${lang}/${corpus} 
  python examples/wmt21/iterative_mining/labse_encode_slurm.py \
    --input-dir=/large_experiments/mmt/wmt21/monolingual_preprocessed/${lang}/${corpus} \
    --output-dir=/large_experiments/mmt/wmt21/labse_embeddings/monolingual/${lang}/${corpus} 
done

