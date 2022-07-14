### Setup
- `git clone git@github.com:fairinternal/fairseq-py.git && cd fairseq-py && git checkout gshard-with-base`
- if you don't have fairseq conda env, follow [these instructions](https://fb.workplace.com/groups/fairseq/permalink/262715387865587/)
- `pip install fairscale`

### Model Paths
```bash
export base_moe_16_gpu=/checkpoint/yuktmitash17/2021-06-29/lm_with_moe_base_16_gpu.fp16.transformer_lm_gpt.shareemb.drop0.1.adam.beta0.9_0.98.wd0.01.clip0.0.lr0.0005.warmup4000.initlr1e-07.sampletok512.breaknone.maxtok2048.updatefreq2.seed2.ngpu16/checkpoint_best.pt
export base_moe_128_gpu=/checkpoint/yuktmitash17/2021-07-14/lm_with_moe_base_16_gpu.fp16.transformer_lm_gpt.shareemb.drop0.1.adam.beta0.9_0.98.wd0.01.clip0.0.lr0.0005.warmup4000.initlr1e-07.sampletok1024.breaknone.maxtok2048.updatefreq8.seed2.ngpu128/checkpoint_best.pt
```

### Eval Command
For the 16 GPU model you only need one node for inference. For the 128 GPU model, you will need at least 4

```bash
source activate fairseq-20210318
export base_moe_16_gpu=/checkpoint/yuktmitash17/2021-06-29/lm_with_moe_base_16_gpu.fp16.transformer_lm_gpt.shareemb.drop0.1.adam.beta0.9_0.98.wd0.01.clip0.0.lr0.0005.warmup4000.initlr1e-07.sampletok512.breaknone.maxtok2048.updatefreq2.seed2.ngpu16/checkpoint_best.pt
sbatch --gres=gpu:1 --nodes=1 --partition=learnfair --time="3-00:00:00" --cpus-per-task 8 --ntasks-per-node 1 --mem=400G --constraint volta32gb --output base_moe_eval_output.out examples/base/example_eval.sh
```
- output will be in `base_moe_eval_output.out`
- results will be in `base_moe_16.json`

### Training
Example BASE LM training command
```bash
python fb_sweep/sweep_lm_wikitext103_transformer_with_base.py --data /private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/ --prefix lm_with_moe_base_16_gpu --num-nodes 2 --num-gpus 8 --num-trials -1 --time 4000 --partition learnfair --constraint="volta32gb"
```