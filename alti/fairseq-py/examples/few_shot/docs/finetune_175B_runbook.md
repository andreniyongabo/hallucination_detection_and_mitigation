### 175B Recipe



TLDR:
- preprocess Flan data, finetune w model parallelism, consolidate checkpoints to 1 file, reshard to 512 files, run few shot eval
- Last updated: Feb 15, 2022

(All below run on azure)

- Finetune on `gshard_combine_megatron_fsdp`

Venv: You will need to be on specific megatron/fairscale versions, but the first time you run fientuning it will print nice instructions.


### Preprocess FLAN Data to jsonl format

the script `examples/few_shot/prompt_to_lm_data.py` just reads the Flan tasks  (using `examples/few_shot/tasks.py`) and converts them to jsonl format and the right file structure.
(see `/private/home/sshleifer/fairseq-py/flan_streaming` for the result).
```
PYTHONPATH='.' python examples/few_shot/prompt_to_lm_data.py --split all --save-dir flan_streaming
```

Result:

```
head -n 1 flan_streaming/valid/00/flan__aeslc_10templates.jsonl

{"text": "This is the content of an email: Elizabeth-  I am working on the Federal Estate Tax Return for your mother's Estate and I notice that I don't have any expenses in my file. Do you have a list of expenses such funeral expenses, etc.? Ellen H. Arthur  Hodes, Ulman, Pessin & Katz, PA  901 Dulaney Valley Road, Suite 400  Towson, MD 21204  (410)769-6146  NOTICE: The information contained in this electronic mail transmission is intended by Hodes, Ulman, Pessin & Katz, P.A. for the use of the named individual or entity to which it is directed and may contain information that is privileged or otherwise confidential. It is not intended for transmission to, or receipt by, anyone other than the named addressee (or a person authorized to deliver it to the named addressee). It should not be copied or forwarded to any unauthorized persons. If you have received this electronic mail transmission in error, please delete it from your system without copying or forwarding it, and notify the sender of the error by reply email or by calling Hodes, Ulman, Pessin & Katz, P.A. at (410) 938-8800 or 1-800-276-0806 so that our address record can be corrected. \nWhat was the subject line for this email? Estate of Joan M. Sager"}
```

### Finetune

```bash
RF175=/data/xlmg/models/175B_ws512/reshard.pt
PYTHONPATH='.' ./fb_sweep/ft_stream.py  -t 1 -n 64 -g 8 -p fts --checkpoints-dir ft_stream2 \
    --lr 1e-5 --tps 2048 --resume-failed  -p doc-block \
    --mp 8 --model-name $RF175 --zero3 --mu 30000 --no-fp16-adam
```

- `train.log`: `/shared/home/sshleifer/fairseq-py/ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu10000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp32adam.lr1e-05.ngpu512/train.log`
- (Here are [Instructions](https://github.com/fairinternal/fairseq-py/pull/3060) to make `/data/xlmg/models/175B_ws512/reshard.pt`.)

Download with azcopy (5 min), find the correct `URL` with `grep https $train_log` and your mouse lol.
```
URL=https://fairacceleastus.blob.core.windows.net/sshleifer/2022-02-14/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512/\?sv\=2020-08-04\&ss\=b\&srt\=sco\&sp\=rwdlactfx\&se\=2023-10-06T11:23:33Z\&st\=2021-10-06T03:23:33Z\&spr\=https\&sig\=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D
mkdir -p ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512
azcopy cp --recursive --include-pattern "*4000*.pt" \
    $URL \
    ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512

# I screwed up a bit...
mv ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512/*.pt ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512/
```
srun
```bash
srun --gres=gpu:8 --nodes=1  --time="2-00:00:00"  --mem=1168000 --pty /bin/bash -l
conda activate fairseq-20210913
```
consolidate mp shards to global checkpoint (1h)
```bash
model_prefix=ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512/checkpoint_3_4000
python scripts/consolidate_fsdp_shards.py $model_prefix --new-arch-name transformer_lm_gpt --save-prefix /data/xlmg/models/175B_flan_4k_sbm_none
```
It should be 368GB

```bash
git checkout gshard
```



reshard global checkpoint to 512 FSDP shards so we can load it faster
```bash
model_prefix=data/xlmg/models/175B_flan_4k_sbm_none
d=/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp29
tok_dir=/data/xlmg/gptz/tokenizers
dst=/data/xlmg/models/resharded_175B_flan_4k_sbm_none
mkdir -p $dst
time python fairseq_cli/reshard.py \
    --path /data/xlmg/models/175B_flan_4k_sbm_none.pt \
    --ddp-backend fully_sharded \
    --vocab-filename $tok_dir/gpt2-vocab.json \
    --merges-filename $tok_dir/gpt2-merges.txt \
    --target-world-size 512 \
    --save-dir $dst \
    --save-prefix checkpoint_3_4000
```

- manually add new path to `examples/few_shot/model_configs.py` with `gptz_sharded_config`

```
AZURE={
    "175B_flan_4k_sbm_none": gptz_sharded_config("/shared/home/sshleifer/fairseq-py/175B_flan_4k_sbm_none/checkpoint_3_4000.pt"),
    ...
}
```
get allocation for evaluation
```
salloc --gres=gpu:8 --partition=hpc --nodes=4  --time="2-00:00:00"   --mem=1168000
```

`srun gpt3_eval`

```bash
export m=175B_flan_4k_sbm_none
export rd=ftstream_azure_results/ftstream_azure_results_v2_"$m"
export FSD=/data/xlmg/few_shot_data
export NCCL_DEBUG="WARN"
mkdir -p $rd
PYTHONPATH="."  srun python -m examples.few_shot.gpt3_eval --max-tokens 2048 \
    --results-dir $rd --batch-size 1 --nshot 0 --model-name $m \
    --tasks sst2 arcchallenge arceasy piqa \
    --distributed-port 15234 \
    --user-dir examples/few_shot/finetune | tee $rd/log.out
```
Note:
- `flan__arc_easy.alltemplates flan__arc_challenge.alltemplates flan__sst2.alltemplates` are very large datasets and take a long time.

### Azcopy results
(From Azure) sync results to FAIR cluster

```
export m=175B_flan_4k_sbm_none
export rd=ftstream_azure_results/ftstream_azure_results_v2_"$m"
DEST_URL=https://fairacceleastus.blob.core.windows.net/sshleifer/175B/ft_eval_results/?sv\=2020-08-04\&ss\=b\&srt\=sco\&sp\=rwdlactfx\&se\=2023-10-06T11:23:33Z\&st\=2021-10-06T03:23:33Z\&spr\=https\&sig\=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D
azcopy sync --recursive $rd $DEST_URL
```


(From Fair Cluster):
```
DEST_URL=https://fairacceleastus.blob.core.windows.net/sshleifer/175B/ft_eval_results/?sv\=2020-08-04\&ss\=b\&srt\=sco\&sp\=rwdlactfx\&se\=2023-10-06T11:23:33Z\&st\=2021-10-06T03:23:33Z\&spr\=https\&sig\=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D
azcopy sync --recursive  $DEST_URL ftstream_azure_results/ftstream_azure_results_v2_175B_flan_4k_sbm_none/
```

### Debug Finetuning Commands

(after srun on fair cluster)
```
PYTHONPATH='.' ./fb_sweep/ft_stream.py  -t 1 -n 8 -g 8 -p dbg --no-save \
    --constraint volta32gb --checkpoints-dir ./gshard_models --model 125M_gpt3_setting \
    --lr 1e-5 --local --max-update 10 --streaming --tps 2048

```
