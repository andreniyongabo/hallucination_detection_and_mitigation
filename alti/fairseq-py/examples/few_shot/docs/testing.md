- last update: July 26, 2021
- author: @sshleifer, @todpole3

# Etiquette
- Run tests before merging PR.
- Add tests to avoid the situation where other people break your code without knowing it.
- If you forgot to run the tests before merging your PR, and your PR broke somebody else's test, it is your job to fix it promptly, either by reverting your change or using your brain.
- If somebody's PR broke something you care about, **they** are doing **you** a favor if they help you resolve. You should #thanks them and then add a test to avoid it happening again.
  - If it's not easy to test your code path, write a PSA in XLM-G-Working Group about how teammates can avoid breaking it.


# Running Existing Tests

```bash
pip install pytest
CUDA_VISIBLE_DEVICES=0,1 pytest tests/gpu/ -sv -k fsdp --maxfail 1
CUDA_VISIBLE_DEVICES=0,1 pytest tests/gpu/ -sv -k moe --maxfail 1
CUDA_VISIBLE_DEVICES=0,1 pytest tests/gpu/test_few_shot.py -sv --maxfail 1
```

Tips:
- if you run all the tests at once they sometimes hang even though nothing is broken. Please help if you know how to fix this!
- you can prefix any invocation with `CUDA_VISIBLE_DEVICES=0,1` to change the world size. Lower usually faster and more reliable.
- You can remove `--sv` to reduce stdout noise.
- All these principles apply on `fairseq:master`, `fairscale` and in many other repos. They are very useful skills to have.


# Adding Tests
## Motivation
Part of our effort to prevent breaking other people's code paths on the `gshard` branch is automated by `pytest`.
You can trade 30 minutes to protect yourself from other people breaking your code path by following these instructions.
- Once you have merged your tests into `gshard`, if another person's PR changes the test from passing to failing ("they broke your test"), it is their responsibility to **promptly** fix the test.
- Upon hearing that they have broken somebody else's test, the offender should apologize for not running the tests before they merged their PR, pause what they are doing and fix the test without your help, either by reverting their change or using their brain to make everyone happy. If they give you trouble send them a link to this post!

After the initial investment, the marginal cost of future proofing new features (like "make sure XX works with FSDP" or make sure XX works with my new command line arg is about 5 minutes. (1 minute if you use the [github cli](https://cli.github.com/manual/index) to automate PR creation.)

### Setup workspace

```bash
git checkout gshard
git pull
# visually check to make sure this worked!
git checkout -b add-fun-new-test
```

## Add training test to `tests/gpu/test_binaries_gpu.py`

#### Create dummy data in the correct format
- See if there is already a dummy_data creator for your task, like [`create_dummy_data`](https://github.com/fairinternal/fairseq-py/blob/gshard/tests/utils.py#L164) which works for LM (monolingual and multilingual) and MT.
- If there isn't, write one.
- Copy/modify logic of `test_resume_training` as needed.



## Add Few Shot Inference Test


### Get a shell with 8 GPUs
```bash
srun --gres=gpu:8 --partition=XLM-G --nodes=1 --time="3-00:00:00" --cpus-per-task 64 --ntasks-per-node 1 --mem=400G --constraint volta32gb --pty /bin/zsh -l
```

Find the sweep script for your task/model, for me this was `sweep_gshard_multilingual_lm.py`
- Edit the sweep script such that it trains a tiny model for 10 steps.
  - `--max-update` from big value to 10
  - We already had a tiny arch so it was easy to set `hyperparam('--arch', 'transformer_lm_gpt2_tiny', save_dir_key=lambda val: val),`
- Train a tiny model on 8 GPUs with `--local`
```bash
MU=10 ./fb_sweep/sweep_gshard_multilingual_lm.py -n 1 -g 8 \
  -d $DATA_DIR
  --local -t 1 -p tiny_multi \
  --checkpoints-dir /checkpoint/sshleifer/
```
- Ensure that it is actually small.
```bash
du -h /checkpoint/sshleifer/tiny_multilingual_2e.me_fp16.bm_none.tps1024.samplealpha3.0.nlangs_1.transformer_lm_gpt2_tiny.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf1.mu10.s1.ngpu8
```
- move it to a nicer path to save screen space for future readers
```bash
ckpt=/checkpoint/$USER
dest=$ckpt/tiny_multi_8e
mv $ckpt/tiny_multi.me_fp16.bm_none.tps1024.samplealpha3.0.nlangs_1.transformer_lm_gpt2_tiny.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf1.mu10.s1.ngpu8 $dest
```
- [Bonus] Make the checkpoint even smaller.Lots of people will wait for this to load in future.
```bash
# remove opt state to make loading faster, this makes new `$dest/checkpoint_eval-{suffix}.pt`
python scripts/remove_opt_state.py $dest/checkpoint_last $dest
echo $dest/checkpoint_eval.pt  # copy this to clipboard
```
- copy model prefix (e.g. `/checkpoint/sshleifer/tiny_multi_8e/checkpoint_eval.pt` (notice lack of `"shared"` even though MOE)) to `examples/few_shot/model_configs.py`
- test from command line
```bash
python -m examples.few_shot.gpt3_eval --model-name tiny_multi_moe \
  --nb-few-shot-samples-values 0 \
 --tasks copa --max-tokens 2048
```
- copy [this test](https://github.com/fairinternal/fairseq-py/blob/gshard/tests/gpu/test_few_shot.py), rename and change model_name
- run the test with `pytest tests/gpu/test_few_shot.py -k tiny_multi`.
- If your test passes in roughly <= 60 seconds, which is slower than we want but the current state of things, you are done. Otherwise, you have to use your brain.



### Re make tiny english models

```bash

train_125m () {
    ./fb_sweep/benchmark_lm.py -g 8 -t 1 -n 1  --dl 2 --embed-dim 256 \
    --bs 4 --li 50 --epg 0 --mu 100 \
    --constraint volta32gb --partition learnaccel \
    --resume-failed --nw 0 -p tiny \
    --checkpoints-dir $ED --opt adam \
    --ddp no_c10d \
    "$@"
}
```


### Few-shot with prompting
```bash
pytest tests/test_few_shot_tasks.py

# make sure that the tasks are running
python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t test_multichoice test_lama test_mlama diagnosis -m 125M_gpt3_setting --nshot 0  -o /checkpoint/$USER/few_shot/debug_multichoice --override-completed --local
```
