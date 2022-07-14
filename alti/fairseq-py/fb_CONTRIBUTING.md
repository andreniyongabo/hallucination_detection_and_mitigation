# FAIR Cluster Enviornment Setup
*on your devfair server:*

## Checkout Repo:
```
cd ~/
git clone -b gshard git@github.com:fairinternal/fairseq-py.git
```

## Set up New Conda
*note: module purge needs to be rerun at the start of a new session (we recommend adding it to your .bashrc file).*
```
module purge && module load anaconda3/2021.05 cudnn/v8.0.3.33-cuda.11.0 cuda/11.1 fairusers_aws nvtop/1.0.0/gcc.7.3.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0 ripgrep/11.0.2 NCCL/2.8.3-1-cuda.11.0
conda create -n fairseq_py38 python=3.8 -y
conda activate fairseq_py38
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


## Install Apex and Megatron-LM deps:
*with conda activated*
*note: this is slow*
```
cd ~/
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 1cd1181dffaac8a6cd92ec117430484d8d4d3fb1
TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6" pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ~/
git clone --depth=1 --branch v2.4 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -r requirements.txt
pip install -e .
cd ~/
```

## Install Fairscale From Source:
*this is needed until a new fairscale release occurs*
```
cd ~/
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout 9f347f373e32ee5cad11a40b70b8e28a74b5e2d4
pip install -e .
cd ~/
```

## Setup Repo
*with conda activated*
```
cd ~/fairseq-py
pip install -e .[dev,few_shot,gpu]
python setup.py build_ext --inplace
```

## Run Test
```
cd ~/fairseq-py/tests
python3 -m pytest --continue-on-collection-errors
```
