### Eval command for english only model

```bash
srun --gres=gpu:8 --partition=XLM-G --nodes=1 --time="3-00:00:00" --cpus-per-task 64 --ntasks-per-node 1 --mem=400G --constraint volta32gb --pty /bin/zsh -l

M="/large_experiments/xlmg/models/sshleifer/expert_drop/shru_a100_longer.dl12.d2048.ngpu8/checkpoint_last_eval.pt"

bash pile/eval_ppl.sh Enron_Emails $M pile_moe8e --distributed-world-size 8 --is-moe

cat pile_moe8e/results.Enron_Emails.json

```



### Eval Commands for multilingual models

```bash
salloc  --gpus-per-node 8 --nodes 2 --ntasks-per-node 1 \
  --cpus-per-task 8 --time 4320 --mem-per-cpu 7G  --constraint volta32gb \
  --partition dev,learnfair
```



Alternatively, you can `srun` just one (model, dataset) combo with

```bash
srun python -m examples.few_shot.eval_pile  moe_128exp_lang16 Enron_Email
```

To run many combos,  edit `few_shot/eval_pile_array.py`, `get_args_to_run` function. It should return two lists of equal lengths, models and datasets.

Then run
```bash
srun python -m examples.few_shot.eval_pile_array 
```

###  Preprocessing (mimicking RoBERTa preprocessing)

```
cd $HOME/ThePile
```

# identify all sets
```bash
cat test.jsonl val.jsonl | extract_json.py meta.pile_set_name | sort | uniq -c > set_counts.txt
cat set_counts.txt | sed "s/^\s*//g" | cut -d ' ' -f 2- > sets.txt
```

# split val/test based on subset
```bash
python split_sets.py --sets sets.txt --output valid --output-dir data val.jsonl
python split_sets.py --sets sets.txt --output test --output-dir data test.jsonl
```

### SPM encode

```bash
export fdir=/private/home/sshleifer/fairseq-py
export SPM_ENCODE=$fdir/pile/spm_encode.sh

find data -name \*.txt -exec $SPM_ENCODE {} \;
```
## binarize with fairseq-preprocess
#### CC100XL
```bash
export DICT=/large_experiments/moe/cc100_xl/bin/shard0/dict.txt
export dest=spm_cc100_xl
mkdir -p $dest
ls data | while read SETPATH; do \
  SETNAME=$(echo $SETPATH | tr -d '/'); \
  echo $SETNAME; \
  fairseq-preprocess --only-source --srcdict $DICT --validpref data/$SETNAME/valid.txt.bpe --testpref data/$SETNAME/test.txt.bpe --destdir $dest/$SETNAME --workers 20
done
```

#### CC100
```bash
export DICT=/datasets01/cc100-bin/072820/250/shard0/dict.txt
export dest=spm_cc
mkdir -p $dest
ls data | while read SETPATH; do \
  SETNAME=$(echo $SETPATH | tr -d '/'); \
  echo $SETNAME; \
  fairseq-preprocess --only-source --srcdict $DICT --validpref data/$SETNAME/valid.txt.bpe --testpref data/$SETNAME/test.txt.bpe --destdir $dest/$SETNAME --workers 20
done
```



#  Old Evaluation steps from Myle


# example eval command (non-model parallel):
FAIRSEQ=/absolute/path/to/fairseq
PYTHONPATH=$FAIRSEQ ./eval_ppl.sh Enron_Emails /path/to/model.pt

# eval on slurm (non-model parallel)
FAIRSEQ=/absolute/path/to/fairseq
MODEL=/private/home/myleott/models/xlmg/unidir_lms/125M/few_shot.roberta+cc100.os.bm_none.tps1024.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms8.uf2.mu572204.s1.ngpu32/checkpoint_best.pt
MODEL=/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu715.dr0.1.atdr0.1.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoint_3_500000.pt
MODEL=/private/home/myleott/models/xlmg/unidir_lms/1.3B/few_shot.roberta+cc100.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_last.pt
MODEL=/private/home/myleott/models/xlmg/unidir_lms/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128/checkpoint_1_100000-shard0.pt
MODEL=/private/home/myleott/models/xlmg/unidir_lms/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128/checkpoint_1_50000-shard0.pt
ls data | while read SETPATH; do \
  SETNAME=$(echo $SETPATH | tr -d '/'); \
  echo $SETNAME; \
  PYTHONPATH=$FAIRSEQ sbatch --gpus=1 --nodes 1 -c 6 --time 1440 --constraint volta32gb --partition learnfair --wrap "./eval_ppl.sh $SETNAME $MODEL"; \
done

# eval (model parallel)
FAIRSEQ=/absolute/path/to/fairseq
MODEL=/private/home/myleott/models/xlmg/unidir_lms/11B/megatron_11b/model.pt
ls data | while read SETPATH; do \
  SETNAME=$(echo $SETPATH | tr -d '/'); \
  echo $SETNAME; \
  PYTHONPATH=$FAIRSEQ sbatch --gpus=8 --nodes 1 -c 40 --time 1440 --constraint volta32gb --partition learnfair --wrap "./eval_ppl.mp.sh $SETNAME $MODEL"; \
done


#
#  Preprocessing steps (mimicking preprocessing from /large_experiments/moe/data/the_pile/binarized/00)
#

PYTHONPATH=/private/home/myleott/src/fairseq10 python bpe_encode_from_jsonl_with_gpt_newlines.py --sets sets.txt --output valid --output-dir data-gpt-newlines val.jsonl
PYTHONPATH=/private/home/myleott/src/fairseq10 python bpe_encode_from_jsonl_with_gpt_newlines.py --sets sets.txt --output test --output-dir data-gpt-newlines test.jsonl

# binarize
ls data | while read SETPATH; do \
  SETNAME=$(echo $SETPATH | tr -d '/'); \
  echo $SETNAME; \
  fairseq-preprocess --only-source --srcdict ~myleott/data/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/dict.txt --validpref data-gpt-newlines/$SETNAME/valid.txt --testpref data-gpt-newlines/$SETNAME/test.txt --destdir data-gpt-newlines-bin/$SETNAME --workers 20
done

#
#  Evaluation steps (mimicking preprocessing from /large_experiments/moe/data/the_pile/binarized/00)
#

~/data/ThePile/data-gpt-newlines-bin/Enron_Emails/

# eval on slurm (non-model parallel)
MODEL=/private/home/myleott/models/xlmg/unidir_lms/1.3B/few_shot.the_pile.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_5_50000.pt
ls data | while read SETPATH; do \
  SETNAME=$(echo $SETPATH | tr -d '/'); \
  echo $SETNAME; \
  sbatch --gpus=1 --nodes 1 -c 6 --time 1440 --constraint volta32gb --partition learnfair --wrap "./eval_ppl_gpt_newlines.sh $SETNAME $MODEL"; \
done
