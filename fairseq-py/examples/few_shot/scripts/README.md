# Available debug scripts

TO DO: Describe debug scripts with 1-2 sentences.

## Notebook setup
Some of the scripts are actually notebooks.
Here are some handy scripts for setting up:

Activate your current env:
```
conda activate fairseq-20210102
```

Install ipykernel and jupyter if you haven't
```
pip install ipykernel
pip install jupyter
```

Add the current environment to jupyter
```
python -m ipykernel install --user --name=fairseq-20210102
```

Activate jupyter
```
jupyter notebook
```

## Scripts

### Few-shot debugging scripts
`examples/few_shot/scripts/debug_task_prompts.ipynb` This script can be used for easily running evaluation with a specific task and prompt and see the generated template. See instructions in the script!