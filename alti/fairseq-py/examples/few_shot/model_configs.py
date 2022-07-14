import copy
import os
import re
from pathlib import Path
from typing import Any, Dict

from examples.few_shot.flan_model_configs import FLAN_MODELS
from examples.few_shot.utils import (
    GPTZ_OVERRIDES,
    IS_AWS,
    IS_AZURE,
    PATH_TO_ROBERTA_DICT,
    dense_bpe_config,
    gptz_sharded_config,
    moe_bpe_config,
)

# Tokenizer files

# SPM BPE model trained on cc100_combined
CC100_COMBINED_SPM = (
    "/large_experiments/xlmg/data/cc100_combined/v1/raw/unsharded/spm_256000.model"
)
CC100_COMBINED_DICT_PATH = (
    "/large_experiments/xlmg/data/cc100_combined/v1/raw/unsharded/dict.txt"
)

# SPM BPE model trained on the new cc100_XL
CC100_XL_BPE_SPM = "/checkpoint/victorialin/cc100_xl_samples/vocabs/cc100_xl_sample_10000000_sp_0_alpha_0.3_256000.0_bpe.model"
CC100_XL_BPE_DICT_PATH = (
    "/large_experiments/xlmg/data/cc100_xl_unigram/bin/shard0/dict.txt"
)
CC100_XL_BPE_DICT_PATH = "/checkpoint/victorialin/cc100_xl_samples/tokenized_cc100_xl_sample_10000000_sp_0_alpha_0.3_256000.0_bpe/dict.txt"

# SPM Unigram model trained on the new CC100_XL
CC100_XL_UNIGRAM_SPM = "/checkpoint/victorialin/cc100_xl_samples/vocabs/cc100_xl_sample_10000000_sp_0_alpha_0.3_256000.0_unigram.model"
CC100_XL_UNIGRAM_DICT_PATH = (
    "/large_experiments/xlmg/data/cc100_xl_unigram/bin/shard0/dict.txt"
)

CC_100_XL_SPM = (
    "/large_experiments/xlmg/data/cc100_combined/v1/raw/unsharded/spm_256000.model"
)

dummy_model = {
    "model_path": "/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/checkpoint_best.pt",
    "dict_path": PATH_TO_ROBERTA_DICT,
    "model_overrides": {"bpe": "gpt2"},
}

OPENAI_API_DUMMY_MODELS = {
    # We use these dummy models to ensure that the generated prompts have the same limitations that we use:
    #  - e.g. 1024 max tokens with gpt2 tokenization, etc.
    "openai_ada": dummy_model,
    "openai_babbage": dummy_model,
    "openai_curie": dummy_model,
    "openai_davinci": dummy_model,
    # Instruction Models
    "openai_ada-instruct-beta": dummy_model,
    "openai_babbage-instruct-beta": dummy_model,
    "openai_curie-instruct-beta-v2": dummy_model,
    "openai_davinci-instruct-beta-v3": dummy_model,
}

HUGGINGFACE_API_DUMMY_MODELS = {
    # We use these dummy models to ensure that the generated prompts have the same limitations that we use:
    #  - e.g. 1024 max tokens with gpt2 tokenization, etc.
    "huggingface_gpt2": dummy_model,
    "huggingface_gpt2-xl": dummy_model,
    "huggingface_EleutherAI=gpt-neo-2.7B": dummy_model,
    "huggingface_bigscience=T0pp": dummy_model,
    "huggingface_bigscience=T0_3B": dummy_model,
}

SSHLEIFER_HOME = "/private/home/sshleifer/fairseq-py"

AZURE = {
    "125M_gpt3_setting": dense_bpe_config(
        "/data/xlmg/models/125M/checkpoint_last-shard0.pt"  
    ),
    "2.7B_gpt3_setting": dense_bpe_config(
        "/data/xlmg/models/2.7B/checkpoint_last-shard0.pt"
    ),
    "6.7B_gpt3_setting": dense_bpe_config(
        "/data/xlmg/models/6.7B/checkpoint_last-shard0.pt"
    ),
    "13B_gpt3_setting": dense_bpe_config(
        "/data/xlmg/models/13B/checkpoint_last_eval.pt"
    ),
    "125M_gptz_reshard": gptz_sharded_config(
        "/shared/home/sshleifer/fairseq-py/125M_gptz_reshard/reshard.pt"
    ),
    "13B_gptz_reshard": gptz_sharded_config(
        "/data/xlmg/models/13B_gptz/resharded/reshard.pt"
    ),
    "175B_flan_4k_sbm_none": gptz_sharded_config(
        "/data/xlmg/models/resharded_175B_flan_4k_sbm_none/checkpoint_3_4000.pt"
    ),
}

AWS = {
    # added by @punit
    "2.7B_gpt3_setting": dense_bpe_config(
        "/fsx/punitkoura/data/xlmg/models/2.7B/checkpoint_last_eval.pt"
    ),
    "6.7B_gpt3_setting": dense_bpe_config(
        "/fsx/punitkoura/data/xlmg/models/6.7B/checkpoint_last-shard0.pt"
    ),
    "13B_gpt3_setting": dense_bpe_config(
        "/fsx/punitkoura/data/xlmg/models/13B/checkpoint_last_eval.pt"
    ),
}

NORMFORMER = {
    "nf_sr_125M_last": dense_bpe_config(
        "/checkpoint/sshleifer/eb/eb_azure/125m_script.stable_emb.fsdp.me_fp16.transformer_lm_gpt.nlay12.emb768.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.003.wu750.dr0.1.atdr0.1.wd0.01.ms2.uf2.mu572204.s1.lat_1.lfc_1.lhead_1.fused.scale_resids.ngpu64/upgraded_checkpoint_last-shard0_eval.pt"
    ),
    "bl_355M_hi_lr": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/355m_script.fsdp.me_fp16.transformer_lm_gpt.nlay24.emb1024.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu750.dr0.0.atdr0.0.wd0.01.ms2.uf1.mu572204.s1.ngpu128/checkpoint_last-shard0_eval.pt"
    ),
    "nf_sr_355M_560": dense_bpe_config(
        "/checkpoint/sshleifer/eb/355m_script.stable_emb.fsdp.me_fp16.transformer_lm_gpt.nlay24.emb1024.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu750.dr0.0.atdr0.0.wd0.01.ms2.uf1.mu572204.s1.lat_1.lfc_1.lhead_1.fused.scale_resids.ngpu128/upgraded_checkpoint_3_560000-shard0_eval.pt"
    ),
    "bl_1.3B_hi_lr_last": dense_bpe_config(
        "/checkpoint/sshleifer/eb/eb_azure/1B_script_azure.fsdp.me_fp16.transformer_lm_gpt.nlay24.emb2048.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.0006.wu375.dr0.1.atdr0.1.wd0.01.ms2.uf2.mu286102.s1.ngpu128/checkpoint_last-shard0_eval.pt"
    ),
    "nf_1B_last": dense_bpe_config(
        "/checkpoint/sshleifer/eb/eb_azure/1B_script_azure_fused.fsdp.me_fp16.transformer_lm_gpt.nlay24.emb2048.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.0006.wu375.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.lat_1.lfc_1.lhead_1.ngpu128/upgraded_checkpoint_last-shard0_eval.pt"
    ),
    # Below here use relu^2 activations
    "bl_2.7B_229": dense_bpe_config(
        f"{SSHLEIFER_HOME}/xlmg.175b_prep.kitchen_sink_baseline.fsdp.me_fp16.zero2.transformer_lm_gpt.relu_squared.nlay32.emb2560.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu375.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_3_229000-shard0_eval.pt"
    ),
    #'ln_2.7B_250': dense_bpe_config(f'{HOME_FS}/ln.fsdp.me_fp16.zero2.transformer_lm_gpt.relu_squared.nlay32.emb2560.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.0006.wu375.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ln_attn.ln_fc.scale_heads.ngpu256/upgraded_checkpoint_3_250000-shard0_eval.pt', ),
    "nf_2.7B_last_v2": dense_bpe_config(
        f"{SSHLEIFER_HOME}/ln.fsdp.me_fp16.zero2.transformer_lm_gpt.relu_squared.nlay32.emb2560.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.0006.wu375.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ln_fc.scale_heads.ngpu256/checkpoint_last-shard0_eval.pt"
    ),
}

DUMMY_MODELS = {
    "random": {  # this will only be used with the random predictor. It is a dummy model because we want to have an easier way to distinguish when displaying results.
        "model_path": "/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/checkpoint_best.pt",
        "dict_path": "/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/dict.txt",
        "model_overrides": {"bpe": "gpt2"},
    },
    "majority": {  # this will only be used with the random predictor. It is a dummy model because we want to have an easier way to distinguish when displaying results.
        "model_path": "/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/checkpoint_best.pt",
        "dict_path": "/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/dict.txt",
        "model_overrides": {"bpe": "gpt2"},
    },
    "sharded_fsdp": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/lm.lr0.001.adam.tps1024.dl12.d768.nh_12.wu500.dr0.1.bs1.ngpu8/checkpoint_1_10.pt",
        fsdp=True,
        use_sharded_state=True,
    ),
}

UNIDIR_LM_ROBERTA_DATA = {
    "125M_gpt3_setting": {
        "model_path": "/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/checkpoint_best.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {"bpe": "gpt2"},
    },
    "355M_gpt3_setting": {
        "model_path": "/checkpoint/sshleifer/gpt3_355m_h1_2021/checkpoint_eval.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {"bpe": "gpt2"},
    },
    # "355M_gpt3_setting_570k_updates": {
    #     # NOTE: this checkpoint stops a bit short of checkpoint_last (586k updates),
    #     #       but is what we had been using prior to 2021-06-18
    #     "model_path": "/large_experiments/xlmg/models/dense/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoint_3_570000.pt",
    #     "dict_path": "/large_experiments/xlmg/models/dense/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64/dict.txt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    "1.3B_gpt3_setting": {
        "model_path": "/large_experiments/xlmg/models/dense/1.3B/few_shot.roberta+cc100.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_last.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {"bpe": "gpt2"},
    },
    # "1.3B_gpt3_setting_thepile": {
    #     # NOTE: ThePile model should not be copied or used for downstream analysis in a paper;
    #     #       please reach out to myleott before using it anywhere
    #     "model_path": "/large_experiments/xlmg/models/dense/1.3B/few_shot.the_pile.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_last.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    "2.7B_gpt3_setting": {
        "model_path": "/large_experiments/xlmg/models/dense/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128/checkpoint_last-shard0.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {"bpe": "gpt2"},
    },
    "2.7B_gpt_bpe_relu2_229k": {
        "model_path": "/large_experiments/xlmg/models/dense/2.7B/experimental/xlmg.175b_prep.kitchen_sink_baseline.fsdp.me_fp16.zero2.transformer_lm_gpt.relu_squared.nlay32.emb2560.bm_none.tps2048.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu375.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_3_229000-shard0_eval.pt",
        "model_overrides": {"bpe": "gpt2"},
    },
    "2.7B_punctsplit_bpe_relu2_229k": {  # gpt-z column 10
        "model_path": "/large_experiments/xlmg/models/dense/2.7B/experimental/gptz.new_data.fsdp.me_fp16.zero2.transformer_lm_gpt.relu_squared.nlay32.emb2560.bm_none.tps2048.punctsplit.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu375.dr0.1.atdr0.1.wd0.1.ms2.uf1.mu286102.s1.ngpu256/checkpoint_26_229000-shard0_eval.pt",
        "model_overrides": GPTZ_OVERRIDES,
    },
    "6.7B_gpt3_setting_1024ctx": {  # formerly called "dense_6.7B"
        # NOTE that this was trained with a sequence length of 1024, compared to 2048 for GPT-3
        # It also was trained without dropout and with a larger batch size (6M) than GPT-3 6.7B.
        "model_path": "/large_experiments/moe/sshleifer/dense_6.7B/checkpoint_eval.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {
            "bpe": "gpt2",
            "world_size": 2,
        },
    },
    "6.7B_gpt3_setting": {
        # NOTE this model matches GPT-3 (i.e., 2048 context length, dropout 0.1, 2M batch size)
        "model_path": "/large_experiments/xlmg/models/dense/6.7B/xlmg_h2_2021.6_7b.fsdp.me_fp16.transformer_lm_gpt.nlay32.emb4096.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.00012.wu187.dr0.1.atdr0.1.wd0.01.ms8.uf1.mu143051.s1.ngpu128/checkpoint_last/checkpoint_last-shard0.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {
            "bpe": "gpt2",
            "world_size": 2,
        },
    },
    "13B_gpt3_setting": {
        "model_path": "/large_experiments/xlmg/models/dense/13B/xlmg_h2_2021.13b.fsdp.me_fp16.transformer_lm_gpt.nlay40.emb5120.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.0001.wu187.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu143051.s1.ngpu256/checkpoint_last_eval.pt",
        "dict_path": PATH_TO_ROBERTA_DICT,
        "model_overrides": {
            "bpe": "gpt2",
            "world_size": 2,
        },
    },
    "125M_gptz_reshard": gptz_sharded_config(
        "/private/home/sshleifer/fairseq-py/gptz_125M_reshard_128/reshard.pt"
    ),
    "125M_gptz_consolidated": {
        "model_path": "/private/home/sshleifer/fairseq-py/gptz_cons/checkpoint_1_2000_consolidated.pt",
        "extra_args": [
            "--memory-efficient-fp16",
        ],
        "model_overrides": GPTZ_OVERRIDES,
    },
    "175B_last_reshard": gptz_sharded_config(
        "/large_experiments/xlmg/models/sshleifer/175B/reshard.pt"
    ),
    "175B_gpt3_setting__last": gptz_sharded_config(  # adding this for backward compatibility with existing experiments and results
        "/large_experiments/xlmg/models/sshleifer/175B/reshard.pt"
    ),
    "175B_135_reshard": gptz_sharded_config(
        "/large_experiments/xlmg/models/sshleifer/reshard_175B_135/175B/reshard.pt"
    ),
    "175B_gpt3_setting__step00135000": gptz_sharded_config(  # adding this for backward compatibility with existing experiments and results
        "/large_experiments/xlmg/models/sshleifer/reshard_175B_135/175B/reshard.pt"
    ),
    "moe_15B": moe_bpe_config(
        # 204M shared params, 64 experts. combined valid PPL: 13.9 (in train.log)
        "/large_experiments/xlmg/models/moe/15B/xlmg.15b.fsdp.me_fp16.transformer_lm_gpt.nlay12.emb768.nexprt512.moe_w0.01.sqrt_world_size.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.0006.wu750.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu572204.s1.ngpu64/checkpoint_last/checkpoint_last.pt",
        moe_eval_capacity_token_fraction=0.25,
        # This moe_eval_capacity_token_fraction is higher than the one in training, but performs better.
        # You can, for example, improve ThePile/DM_Mathematics PPL from 12.06 to 11.24 by setting moe_eval_capacity_token_fraction=0.5
    ),
    "moe_52B": moe_bpe_config(
        # Same architecture and training settings as 355M_gpt3_setting
        "/large_experiments/xlmg/models/moe/52B/xlmg.52b.fp16.bm_none.tps2048.transformer_lm_gpt2_bigger.dl24.demb1024.dffn4096.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.sqrt_world_size.wu715.dr0.0.atdr0.0.wd0.01.ms2.uf1.mu572204.s1.ngpu128/checkpoint_last_eval/checkpoint_eval.pt",
        # capacity factor during training: 2 * math.ceil(local_bsz_in_tokens / global_num_experts) = 2 * 4096 / 512 = 16
        # capacity factor during eval, assuming a local bsz of 1 x 2048 tokens, then moe_eval_capacity_token_fraction = 16 / 2048 = 0.0078125
        moe_eval_capacity_token_fraction=0.05,
    ),
    "moe_207B": moe_bpe_config(
        # Same architecture and training settings as 1.3B_gpt3_setting
        "/large_experiments/xlmg/models/moe/207B/xlmg.adam_fp16.me_fp16.bm_none.tps2048.transformer_lm_gpt2_bigger.dl24.demb2048.dffn8192.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.sqrt_world_size.wu2000.dr0.1.atdr0.1.wd0.0.ms2.uf1.mu286102.s1.ngpu256/checkpoint_last_eval/checkpoint_eval.pt",
        # capacity factor during training: 2 * math.ceil(local_bsz_in_tokens / global_num_experts) = 2 * 4096 / 512 = 16
        # capacity factor during eval, assuming a local bsz of 1 x 2048 tokens, then moe_eval_capacity_token_fraction = 16 / 2048 = 0.0078125
        moe_eval_capacity_token_fraction=0.05,
    ),
    "moe_523B": moe_bpe_config(  # formerly called "moe_500B_300b_tokens"
        # NOTE this doesn't match any of the GPT-3 architectures
        # NOTE that this was trained with a sequence length of 1024
        # valid_ppl: 4.77 in logs, 4.51 with moe_eval_capacity_token_fraction=0.05
        # formerly located at /checkpoint/sshleifer/moe/moe_500B_300B_tokens/checkpoint_eval.pt
        "/large_experiments/xlmg/models/moe/523B/moe_500b_24l_sqrt.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl24.demb2304.dffn9216.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0007.wu2000.dr0.0.atdr0.0.wd0.0.ms8.uf1.mu72000.s1.ngpu512/checkpoint_last_eval/checkpoint_eval.pt",
        # capacity factor during training: 2 * math.ceil(local_bsz_in_tokens / global_num_experts) = 2 * 8192 / 1024 = 16
        # capacity factor during eval, assuming a local bsz of 1 x 1024 tokens, then moe_eval_capacity_token_fraction = 16 / 1024 = 0.015625
        moe_eval_capacity_token_fraction=0.05,
    ),
    "moe_1.1T": moe_bpe_config(
        # Same architecture and training settings as 6.7B_gpt3_setting_1024ctx
        # NOTE that this was trained with a sequence length of 1024
        # valid_ppl: 4.48 in logs, 4.32 with moe_eval_capacity_token_fraction=0.05
        # formerly located at /large_experiments/xlmg/models/sshleifer/1.1T_skinny/checkpoint_eval.pt
        "/large_experiments/xlmg/models/moe/1.1T/xlmg.1.1T.adam_fp16.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl32.demb4096.dffn16384.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.sqrt_world_size.wu2000.dr0.0.atdr0.0.wd0.0.ms12.uf1.mu47683.s1.ngpu512/checkpoint_last_eval/checkpoint_eval.pt",
        # capacity factor during training: 2 * math.ceil(local_bsz_in_tokens / global_num_experts) = 2 * 12288 / 512 = 48
        # capacity factor during eval, assuming a local bsz of 1 x 1024 tokens, then moe_eval_capacity_token_fraction = 48 / 1024 = 0.046875
        moe_eval_capacity_token_fraction=0.05,
    ),
    "moe_1.1T_2048ctx": moe_bpe_config(
        # Same architecture and training settings as 6.7B_gpt3_setting
        # NOTE that this is the 1.1T MoE model retrained in 09/2021; this was trained with a sequence length of 2048
        # valid_ppl: 4.48 in logs, 4.32 with moe_eval_capacity_token_fraction=0.05
        "/large_experiments/xlmg/models/moe/1.1T/xlmg.1.1T_moe_xlmg_h2_2021.fsdp.me_fp16.zero2.transformer_lm_gpt.nlay32.emb4096.nexprt512.moe_w0.01.sqrt_world_size.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.00012.wu187.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu143051.s1.ngpu256/checkpoint_last_eval/checkpoint_eval.pt",
        # capacity factor during training: 2 * math.ceil(local_bsz_in_tokens / global_num_experts) = 2 * 12288 / 512 = 48
        # capacity factor during eval, assuming a local bsz of 1 x 2048 tokens, then moe_eval_capacity_token_fraction = 48 / 2048 = 0.0234375
        moe_eval_capacity_token_fraction=0.0234375,
    ),
    "tiny_dense": dense_bpe_config(
        "/checkpoint/sshleifer/tiny/dense.lr0.003.adam.tps1024.dl2.d256.wu_500.dr0.0.bs8.ngpu8/checkpoint_last_eval.pt"
    ),
    "tiny_moe": moe_bpe_config(
        "/checkpoint/sshleifer/tiny/moe.8e.lr0.003.adam.tps1024.dl2.d256.wu_500.dr0.0.bs8.ngpu8/checkpoint_eval.pt"
    ),
    "ccnet_head_200k_32gpus": {
        "model_path": "/checkpoint/shuohui/moe_lm/top2_64e/top2_64e.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu200000.s1.ngpu32/checkpoint_last.pt",
        "dict_path": "/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/dict.txt",
        "extra_args": ["--is-moe"],
        "model_overrides": {"bpe": "gpt2", "moe_eval_capacity_token_fraction": 0.05},
    },
    "ccnet_head_200k_64gpus": {
        "model_path": "/checkpoint/shuohui/moe_lm/top2_64e/top2_64e.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu200000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": "/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/dict.txt",
        "extra_args": ["--is-moe"],
        "model_overrides": {"bpe": "gpt2", "moe_eval_capacity_token_fraction": 0.05},
    },
    "c4_28k_32gpus": {
        "model_path": "/checkpoint/shuohui/moe_lm/top2_64e/top2_64e.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu28000.s1.ngpu32/checkpoint_last.pt",
        "dict_path": "/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/dict.txt",
        "extra_args": ["--is-moe"],
        "model_overrides": {"bpe": "gpt2", "moe_eval_capacity_token_fraction": 0.05},
    },
    "c4_100k_32gpus": {
        "model_path": "/checkpoint/shuohui/moe_lm/top2_64e/top2_64e.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu100000.s1.ngpu32/checkpoint_last.pt",
        "dict_path": "/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/dict.txt",
        "extra_args": ["--is-moe"],
        "model_overrides": {"bpe": "gpt2", "moe_eval_capacity_token_fraction": 0.05},
    },
    # Distilled Models
    "64e_student": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/m2d.lr0.003.t_moe_64e_longer_170K.a0.75.temp1.0.wt_moe_64e_longer_170000_eval.dl12.d2048.wu2000.dr0.0.ngpu64/checkpoint_last-rank-0.pt"
    ),
    "1.3B_student": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/scale_soft_loss5x.poly.ebs512.lr0.003.t_1.3B_gpt3_setting.a0.75.temp1.0.wt_alternating_1.3B_layers.dl12.d2048.wu2000.dr0.0.ngpu32/checkpoint_last.pt"
    ),
    "2.7B_student": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/d2d.poly.ebs512.lr0.003.t_2.7B_gpt3_setting.a0.75.temp1.0.wt_alternating_1.3B_layers.dl12.d2048.wu2000.dr0.0.ngpu128/checkpoint_last.pt"
    ),
    "6.7B_student": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/d2d.poly.ebs512.v3.fsdp.bs2.lr0.003.t_6.7B_gpt3_setting_1024ctx.a0.75.temp1.0.wt_alternating_1.3B_layers.dl12.d2048.wu2000.dr0.0.ngpu128/checkpoint_last-shard0.pt"
    ),
    "small_1.3B_student": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/dld.fsdp.poly.ebs512.lr0.003.t_1.3B_gpt3_setting.a0.75.temp1.0.dl12.d768.wu2000.dr0.0.ngpu128/checkpoint_last-shard0.pt"
    ),
    "dense_baseline_709M_300": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/dense.ebs512.lr0.006.dl12.d2048.dr0.0.ngpu64/checkpoint_57_300000-shard0.pt"
    ),
    "dense_baseline_709M_600": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/dense.ebs512.lr0.006.dl12.d2048.dr0.0.ngpu64/checkpoint_114_600000-shard0.pt"
    ),
    "dense_baseline_709M_1000": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/dense.ebs512.lr0.006.dl12.d2048.dr0.0.ngpu64/checkpoint_last-shard0.pt"
    ),
    "125M_tps1024": dense_bpe_config(
        "/private/home/sshleifer/fairseq-py/distill/distill_aws/dld.poly.lr0.003.dl12.d768.wu2000.dr0.0.ngpu64/checkpoint_best.pt"
    ),
    ### Older models that are not recommended:
    # "dense_256e": {
    #     "model_path": "/large_experiments/moe/shru/moe_lm/dense_baseline_256e/dense_baseline_256e.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.001.wu2000.dr0.0.atdr0.0.wd0.0.mt2048.uf8.mu128000.s1.ngpu64/checkpoint_last.pt",
    # },
    # "moe_256e": moe_bpe_config("/large_experiments/moe/shru/moe_lm/top2_256e_sv/top2_256e_sv.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu72000.s1.ngpu64/checkpoint_last.pt"),
    # "moe_264B": moe_bpe_config("/large_experiments/moe/shru/moe_lm/a100s_512e/a100s_512e_v2.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl40.demb1792.dffn7168.moe_w0.01.all.sqrt_world_size.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0007.wu2000.dr0.0.atdr0.0.wd0.0.ms16.uf1.mu72000.s1.ngpu256_converted/checkpoint_1_21500.pt"),
    # "moe_500B": moe_bpe_config("/large_experiments/moe/namangoyal/checkpoints/moe_lms/enlm/moe_500b_24l_sqrt.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl24.demb2304.dffn9216.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0007.wu2000.dr0.0.atdr0.0.wd0.0.ms8.uf1.mu72000.s1.ngpu512/converted/checkpoint_2_40500.pt"),
    # "124M": {
    #     "model_path": "/private/home/myleott/models/public_models/LM/roberta_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu3000.dr0.1.atdr0.1.wd0.01.ms8.uf4.mu100000.s1.ngpu16/model.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "354M": {
    #     "model_path": "/private/home/myleott/models/public_models/LM/roberta_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.003.wu3000.dr0.1.atdr0.1.wd0.01.ms2.uf4.mu100000.s1.ngpu64/model.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "1.5B": {
    #     "model_path": "/private/home/myleott/models/public_models/LM/roberta_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big.share.adafactor.lr0.0015.wu3000.wd0.01.dr0.1.atdr0.1.ms1.uf2.mu100000.s1.ngpu256/model.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "1.5B_more_data": {
    #     "model_path": "/private/home/myleott/models/public_models/LM/roberta_plus_more_data_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_mp_friendly.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0015.wu3000.wd0.01.dr0.1.atdr0.1.ctb.ms16.uf1.mu100000.s1.ngpu256/model.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "11B": {
    #     "model_path": "/checkpoint/myleott/s3/models/model_parallel/megatron_11b/model.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    #     # NOTE: we don't need model parallel here, inference should work on a 32GB V100
    #     # "enabled": torch.cuda.device_count() == 8,
    #     # "model_parallel_args": [
    #     #    "--model-parallel-size", "8",
    #     #    "--distributed-world-size", "8",
    #     # ],
    # },
    # "seq2seq": {
    #     "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.os.bm_none.tps2048.bart_base.seq2seq_lm.share.adam.b2_0.98.eps1e-08.cl1.0.lr5e-05.wu715.dr0.0.atdr0.1.wd0.01.ms4.uf1.mu572204.s1.min_enc_pct0.8.max_enc_pct0.8.ngpu64/model.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "seq2seq_half": {
    #     "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.os.bm_none.tps2048.bart_base.seq2seq_lm.share.adam.b2_0.98.eps1e-08.cl1.0.lr5e-05.wu715.dr0.0.atdr0.1.wd0.01.ms4.uf1.mu572204.s1.min_enc_pct0.8.max_enc_pct0.8.ngpu64/model_half.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "routing_transformer_2048": {
    #     "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.rt.fp16.tps2048.routing_transformer.aux_cross_entropy.adam.cl0.25.cosine.lr0.00025.s2.ngpu64/model.pt",
    #     "extra_args": [
    #         "--user-dir",
    #         "examples/routing_transformer_lm",
    #     ],
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "routing_transformer_8192": {
    #     "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.rt.fp16.routing_transformer.aux_cross_entropy.adam.cl0.25.cosine.lr0.00025.s2.ngpu64/model.pt",
    #     "extra_args": [
    #         "--user-dir",
    #         "examples/routing_transformer_lm",
    #     ],
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "gpt3_355m_nodrop": {
    #     "model_path": "/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoint",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
    # "gpt3_355m_drop": {
    #     "model_path": "/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu715.dr0.1.atdr0.1.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoint_best.pt",
    #     "model_overrides": {"bpe": "gpt2"},
    # },
}

UNIDIR_LM_ROBERTA_DATA.update(NORMFORMER)
if IS_AZURE:
    UNIDIR_LM_ROBERTA_DATA.update(AZURE)
if IS_AWS:
    UNIDIR_LM_ROBERTA_DATA.update(AWS)

UNIDIR_LM_PILE_DATA = {
    "moe_128exp_newdata": {
        "model_path": "/large_experiments/moe/namangoyal/checkpoints/moe_lms/the_pile/the_pile.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu35000.s1.ngpu128/checkpoint_7_35000.pt",
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {"world_size": 16, "bpe": "gpt2"},
    },
    "1.3B_pile": {
        "model_path": "/private/home/myleott/models/xlmg/unidir_lms/1.3B/few_shot.the_pile.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_23_270000.pt",
        "model_overrides": {
            "bpe": "gpt2",
        },
    },
}

MULTI_LM_CC100_DATA = {
    "multilingual_seq2seq": {
        "model_path": "/checkpoint/artetxe/xlmg/run4/xlmg.cc5.os.langs_en_XX_fr_XX_es_XX_it_IT_de_DE.alpha_0.3.bm_none.tps1024.bart_large.seq2seq_lm.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms2.uf8.mu572204.s1.ngpu32/checkpoint_best.pt",
        "dict_path": "/datasets01/cc100-bin/072820/250/shard0/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": "/private/home/louismartin/ts/resources/models/mbart/sentence.bpe.model",
        },
    },
}

MULTI_LM_OLD_CC100_XL_DATA = {
    "dense_lang16": {  # 100000
        "model_path": "/large_experiments/moe/shru/moe_multi_lm/dense/dense.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_16.transformer_lm_gpt2_big_wide.dl12.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu100000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "model_overrides": {
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "dense_lang16_with_bos": {  # 100000
        "model_path": "/large_experiments/moe/shru/moe_multi_lm/dense/dense.me_fp16.bm_none.tps1024.samplealpha0.2.with_bos.nlangs_16.transformer_lm_gpt2_big_wide.dl12.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu100000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "tiny_multi_moe": {
        "model_path": "/checkpoint/sshleifer/tiny_multi_8e/checkpoint_eval.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang16": {  # 58000
        "model_path": "/large_experiments/moe/shru/moe_multi_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_16.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu72000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang16_with_bos": {  # 72000
        "model_path": "/large_experiments/moe/shru/moe_multi_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.with_bos.nlangs_16.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu72000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang4_cc_xl": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_4.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.0.mu512000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,fr_XX,ur_PK,zh_CN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang16_cc_xl": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_16.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.0.mu512000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang32_cc_xl": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.0.mu512000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW,\
                      id_ID,no_XX,hu_HU,nl_XX,sk_SK,he_IL,cs_CZ,lt_LT,ca_ES,sl_SI,ms_MY,ta_IN,tl_XX,eu_ES,te_IN,mr_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang64_cc_xl": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_64.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.0.mu512000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW,\
                      id_ID,no_XX,hu_HU,nl_XX,sk_SK,he_IL,cs_CZ,lt_LT,ca_ES,sl_SI,ms_MY,ta_IN,tl_XX,eu_ES,te_IN,mr_IN,\
                      fa_IR,pt_XX,fi_FI,lv_LV,sq_AL,et_EE,sr_RS,az_AZ,ja_XX,bn_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
                      cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_128exp_lang100_cc_xl": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/top2_128e/top2_128e.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_100.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.0.mu512000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,zh_TW,\
                      id_ID,no_XX,hu_HU,nl_XX,sk_SK,he_IL,cs_CZ,lt_LT,ca_ES,sl_SI,ms_MY,ta_IN,tl_XX,eu_ES,te_IN,mr_IN,\
                      fa_IR,pt_XX,fi_FI,lv_LV,sq_AL,et_EE,sr_RS,az_AZ,ja_XX,bn_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
                      cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM,\
                      ro_RO,da_DK,pl_PL,ko_KR,it_IT,hr_HR,is_IS,gl_ES,kk_KZ,ka_GE,mk_MK,hy_AM,la_VA,be_BY,ml_IN,am_ET,\
                      pa_IN,ku_TR,so_SO,ha_NG,my_MM_zaw,sd_PK,te_IN_rom,km_KH,or_IN,ta_IN_rom,fy_NL,mg_MG,bs_BA,xh_ZA,su_ID,om_KE,\
                      uk_UA,as_IN,yo_NG,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
}

MULTI_LM_CC100_COMBINED_ROBERTA = {
    "l28_dense_1b24_2cc100_combined_roberta": {  # 82000
        "model_path": "/checkpoint/xianl/multilingual_moe_lm/l28_dense.me_fp16.bm_none.tps1024.samplealpha1.nlangs_28.transformer_lm_gpt2_big_wide.dl24.emb1024.mha16.ffn16384.share.adam.b2_0.98.eps1e-08.cl1.0.lr0.0015.wu4000.dr0.1.wd0.0.ms2.uf8.mu100000.s1.ngpu64/checkpoint_best-shard0.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "model_overrides": {
            "langs": "en_XX,et_EE,ht_HT,id_ID,it_IT,qu_PE,sw_KE,ta_IN,th_TH,tr_TR,vi_VN,zh_CN,ar_AR,bg_BG,de_DE,es_XX,fr_XX,hr_HR,hu_HU,lt_LT,mk_MK,pl_PL,pt_XX,sq_AL,sr_RS,el_GR,ru_RU,hi_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "l28_dense_3b48_2cc100_combined_roberta": {  # 100000
        "model_path": "/checkpoint/xianl/multilingual_moe_lm/l28_dense_subs.as.me_fp16.ca.ss.bm_none.tps1024.samplealpha1.nlangs_28.transformer_lm_gpt2_big_wide.dl48.emb2048.mha16.ffn8192.share.adam.b2_0.98.eps1e-08.cl1.0.lr0.0015.wu4000.dr0.1.wd0.0.ms2.uf8.mu100000.s1.ngpu64/checkpoint_17_100000_consolidated.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "model_overrides": {
            "langs": "en_XX,et_EE,ht_HT,id_ID,it_IT,qu_PE,sw_KE,ta_IN,th_TH,tr_TR,vi_VN,zh_CN,ar_AR,bg_BG,de_DE,es_XX,fr_XX,hr_HR,hu_HU,lt_LT,mk_MK,pl_PL,pt_XX,sq_AL,sr_RS,el_GR,ru_RU,hi_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang1_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l1_64e_top2/l1_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_1.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang4_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l4_64e_top2/l4_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_4.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,fr_XX,ur_PK,zh_CN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang64_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l64_64e_top2/l64_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_64.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE,\
                      fa_IR,sk_SK,ms_MY,lv_LV,az_AZ,tl_XX,ja_XX,bn_IN,eu_ES,te_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
                      cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang100_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l100_64e_top2/l100_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_100.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE,\
                      fa_IR,sk_SK,ms_MY,lv_LV,az_AZ,tl_XX,ja_XX,bn_IN,eu_ES,te_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
                      cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM,\
                      no_XX,da_DK,fi_FI,ko_KR,nl_XX,he_IL,cs_CZ,is_IS,gl_ES,kk_KZ,ka_GE,hy_AM,la_VA,be_BY,ml_IN,zh_TW,\
                      mr_IN,am_ET,pa_IN,ku_TR,so_SO,ha_NG,my_MM_zaw,sd_PK,te_IN_rom,km_KH,or_IN,ta_IN_rom,fy_NL,mg_MG,bs_BA,xh_ZA,\
                      su_ID,om_KE,uk_UA,as_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_256exp_lang4_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l4_256e_top2/l4_256e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_4.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu128/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,fr_XX,ur_PK,zh_CN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_256exp_lang32_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_256e_top2/l32_256e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu128/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_256exp_lang64_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l64_256e_top2/l64_256e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_64.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu128/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE,\
                      fa_IR,sk_SK,ms_MY,lv_LV,az_AZ,tl_XX,ja_XX,bn_IN,eu_ES,te_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
                      cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_256exp_lang100_cc100_combined_roberta": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l100_256e_top2/l100_256e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_100.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu128/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE,\
                      fa_IR,sk_SK,ms_MY,lv_LV,az_AZ,tl_XX,ja_XX,bn_IN,eu_ES,te_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
                      cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM,\
                      no_XX,da_DK,fi_FI,ko_KR,nl_XX,he_IL,cs_CZ,is_IS,gl_ES,kk_KZ,ka_GE,hy_AM,la_VA,be_BY,ml_IN,zh_TW,\
                      mr_IN,am_ET,pa_IN,ku_TR,so_SO,ha_NG,my_MM_zaw,sd_PK,te_IN_rom,km_KH,or_IN,ta_IN_rom,fy_NL,mg_MG,bs_BA,xh_ZA,\
                      su_ID,om_KE,uk_UA,as_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
}


MULTI_LM_CC100_COMBINED_ROBERTA_ALPHA_SWEEP = {
    "moe_64exp_lang32_cc100_combined_roberta_0.7": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_1.0": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_28_96000.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_1.0_adapt": {  # POC: Victoria
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2_adaptation/l32_64e_top2_adaptation.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.bl1.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64.32k/checkpoint_59_96000.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_2.0": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha2.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_3.0": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha3.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_4.0": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha4.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_4.85": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha4.85.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
}

MULTI_LM_CC100_XL_300_SHARD_DATA = {
    "dense_lang134_new_cc100_xl_unigram": {  # 100000
        "model_path": "/checkpoint/victorialin/multilingual_dense_lm/cc100_xl_unigram_l134_dense_top2/cc100_xl_unigram_l134_dense_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_134.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoints.40k/checkpoint_14_40000.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_lang134_new_cc100_xl_unigram_seqlen1024_eps1e-8": {  # 100000
        "model_path": "/checkpoint/victorialin/multilingual_dense_lm/cc100_xl_unilm_l134_dense_top2/cc100_xl_unilm_l134_dense_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_134.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu100000.s1.ngpu64/checkpoint_17_100000.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_lang134_new_cc100_xl_unigram_seqlen1024_eps1e-6": {  # 100000
        "model_path": "/checkpoint/victorialin/multilingual_dense_lm/cc100_xl_unilm_l134_dense_top2/cc100_xl_unilm_l134_dense_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_134.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-06.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms4.uf4.mu100000.s1.ngpu64/checkpoint_last_0912.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_lang134_new_cc100_xl_old_bpe": {  # 100000
        "model_path": "/checkpoint/victorialin/multilingual_dense_lm/cc100_xl_combined_vocab_l134_dense_top2/cc100_xl_combined_vocab_l134_dense_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_134.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoints.40k/checkpoint_14_40000.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang134_new_cc100_xl_unigram": {  # 100000
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/cc100_xl_unigram_l134_64e_top2/cc100_xl_unigram_l134_64e_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_134.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_9_25000.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "moe_64exp_lang134_new_cc100_xl_old_bpe": {  # 100000
        "model_path": "",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
}

MULTI_LM_CC100_XL_SUPER_SHARD_DATA = {
    "dense_564M_lang30_new_cc100_xl_unigram__step119209": {  # 119209
        "model_path": "/large_experiments/xlmg/models/multilingual/dense/564M/checkpoint_last-shard0.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_1.7B_lang30_new_cc100_xl_unigram__step58000": {  # 100000
        "model_path": "/large_experiments/xlmg/models/multilingual/dense/1.7B/checkpoint_22_58000-shard0.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_2.9B_lang30_new_cc100_xl_unigram__step59604": {  # 59604
        "model_path": "/large_experiments/xlmg/models/multilingual/dense/2.9B/checkpoint_last-shard0.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_7.5B_lang30_new_cc100_xl_unigram__step00065000": {  # 100000
        "model_path": "/large_experiments/xlmg/models/multilingual/dense/7.5B/checkpoint_6_65000-shard0.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_7.5B_lang30_new_cc100_xl_unigram__step238000": {  # 100000
        "model_path": "/large_experiments/xlmg/models/multilingual/dense/7.5B/checkpoint_last-shard0.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_lang134_cc100_xl_supershard_unigram_alpha_1.0": {  # 100000
        "model_path": "/large_experiments/nllb/moe/namangoyal/checkpoints/moe_lms/moe_xlmr/dense_unilm_cc100_xl_unilm_supershards.me_fp16.siu10000.s1.0.cmpltdoc.tps1024.transformer_lm_gpt.nlay24.emb1024.nlangs_134.adam.b2_0.98.eps1e-06.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms4.uf4.mu100000.s1.ngpu64/checkpoint_6_100000.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_lang134_cc100_xl_supershard_unigram_alpha_0.7": {  # 100000
        "model_path": "/large_experiments/nllb/moe/namangoyal/checkpoints/moe_lms/moe_xlmr/dense_unilm_cc100_xl_unilm_supershards.me_fp16.siu10000.s0.7.cmpltdoc.tps1024.transformer_lm_gpt.nlay24.emb1024.nlangs_134.adam.b2_0.98.eps1e-06.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms4.uf4.mu100000.s1.ngpu64/checkpoint_6_100000.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "dense_lang134_cc100_xl_supershard_unigram_alpha_0.3": {  # 100000
        "model_path": "/large_experiments/nllb/moe/namangoyal/checkpoints/moe_lms/moe_xlmr/dense_unilm_cc100_xl_unilm_supershards.me_fp16.siu10000.s0.3.cmpltdoc.tps1024.transformer_lm_gpt.nlay24.emb1024.nlangs_134.adam.b2_0.98.eps1e-06.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms4.uf4.mu100000.s1.ngpu64/checkpoint_6_100000.pt",
        "dict_path": CC100_XL_UNIGRAM_DICT_PATH,
        "model_overrides": {
            "langs": "af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
        },
    },
    "moe_200B_lang30_new_cc100_xl_unigram__step00048000": {  # 48000
        "model_path": "/large_experiments/xlmg/models/multilingual/moe/200B/200b.fsdp.zero2.me_fp16.transformer_lm_gpt2_big_wide.nlay48.emb2048.nexprt256.moe_w0.01.top1.sqrt.nlangs30.alpha1.0.bm_none.tps2048.stable.blockwise.adam8bit.fp16adam.b2_0.98.eps1e-08.cl1.0.lr0.00012.wu4000.dr0.1.atdr0.1.wd0.0.ms4.uf4.mu119209.s2.ngpu128/checkpoint_13_48000/checkpoint_13_48000_consolidated.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        # capacity factor during training: math.ceil(local_bsz_in_tokens / global_num_experts) = (2048 * 4 / 256) = 32
        # capacity factor during eval, assuming a local bsz of 1 x 2048 tokens, then moe_eval_capacity_token_fraction = 32 / 2048 = 0.015625
        "model_overrides": {
            "world_size": 32,
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
            "moe_eval_capacity_token_fraction": 0.015625,
        },
    },
    "moe_500B_lang30_new_cc100_xl_unigram__step00017000": {  # 17000
        "model_path": "/large_experiments/xlmg/models/xianl/multilingual_big_run/moe/500B/1t.fsdp.zero2.me_fp16.transformer_lm_gpt2_big_wide.nlay32.emb4096.nexprt256.moe_w0.01.top1.sqrt.nlangs30.alpha1.0.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl1.0.lr0.00012.wu187.dr0.1.atdr0.1.wd0.01.ms2.uf2.mu238418.s2.ngpu256/checkpoint_2_17000.pt",
        "dict_path": "/large_experiments/xlmg/data/cc100_xl_unigram/intermediate_bin/shard0/en_XX/dict.txt",
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        # capacity factor during training: 2 * math.ceil(local_bsz_in_tokens / global_num_experts) = 2 * (2048 * 4 / 256) = 64
        # capacity factor during eval, assuming a local bsz of 1 x 2048 tokens, then moe_eval_capacity_token_fraction = 64 / 2048 = 0.03125
        "model_overrides": {
            "world_size": 64,
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_XL_UNIGRAM_SPM,
            "moe_eval_capacity_token_fraction": 0.03125,
        },
    },
}

MULTI_LM_CONTINUOUS_TRAINING_EXPERIMENTS = {
    "ct_moe_64exp_lang32_cc100_combined_roberta_1.0_64000": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_19_66000.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "ct_moe_64exp_lang32_cc100_combined_roberta_1.0_96000": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_28_96000.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "ct_moe_64exp_lang32_cc100_combined_roberta_1.0_adapt": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2_adaptation/l32_64e_top2_adaptation.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.bl1.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64.32k/checkpoint_59_96000.pt",
        "dict_path": CC100_COMBINED_DICT_PATH,
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
                      id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
}


CC_100_XL_SPM_EXPERIMENTAL_MULTILINGUAL_RUN = {
    "l28_dense_1b24_2cc100_combined_roberta": {  # POC: Xian
        "model_path": "/checkpoint/xianl/multilingual_moe_lm/l28_dense.me_fp16.bm_none.tps1024.samplealpha1.nlangs_28.transformer_lm_gpt2_big_wide.dl24.emb1024.mha16.ffn16384.share.adam.b2_0.98.eps1e-08.cl1.0.lr0.0015.wu4000.dr0.1.wd0.0.ms2.uf8.mu100000.s1.ngpu64/checkpoint_best-shard0.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "model_overrides": {
            "langs": "en_XX,et_EE,ht_HT,id_ID,it_IT,qu_PE,sw_KE,ta_IN,th_TH,tr_TR,vi_VN,zh_CN,ar_AR,bg_BG,de_DE,es_XX,fr_XX,hr_HR,hu_HU,lt_LT,mk_MK,pl_PL,pt_XX,sq_AL,sr_RS,el_GR,ru_RU,hi_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC_100_XL_SPM,
        },
    },
    "l28_64e_1b24_top1_cc100_combined_roberta": {  # POC: Xian
        "model_path": "/checkpoint/xianl/multilingual_moe_lm/l28hi_64e.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_28.transformer_lm_gpt2_big_wide.dl24.emb1024.mha16.moe_w0.01.top1.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX,et_EE,ht_HT,id_ID,it_IT,qu_PE,sw_KE,ta_IN,th_TH,tr_TR,vi_VN,zh_CN,ar_AR,bg_BG,de_DE,es_XX,fr_XX,hr_HR,hu_HU,lt_LT,mk_MK,pl_PL,pt_XX,sq_AL,sr_RS,el_GR,ru_RU,hi_IN",
            "bpe": "sentencepiece",
            "moe_eval_capacity_token_fraction": 0.046875,
            "sentencepiece_model": CC_100_XL_SPM,
        },
    },
    "l28hi_64e_top2_l24": {  # POC: Xian
        "model_path": "/checkpoint/xianl/multilingual_moe_lm/l28hi_64e_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_28.transformer_lm_gpt2_big_wide.dl24.emb1024.mha16.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_best.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 64,
            "langs": "en_XX,et_EE,ht_HT,id_ID,it_IT,qu_PE,sw_KE,ta_IN,th_TH,tr_TR,vi_VN,zh_CN,ar_AR,bg_BG,de_DE,es_XX,fr_XX,hr_HR,hu_HU,lt_LT,mk_MK,pl_PL,pt_XX,sq_AL,sr_RS,el_GR,ru_RU,hi_IN",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC_100_XL_SPM,
            "moe_eval_capacity_token_fraction": 0.046875,
        },
    },
    "moe_64exp_lang1_cc100_combined_roberta_0921": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l1_64e_top2/l1_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_1.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_last.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "extra_args": [
            "--batch-size",
            "2",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 16,
            "langs": "en_XX",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_1.0_0921": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2/l32_64e_top2.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/checkpoint_28_96000.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
    "moe_64exp_lang32_cc100_combined_roberta_1.0_adapt_0921": {
        "model_path": "/checkpoint/victorialin/multilingual_moe_lm/l32_64e_top2_adaptation/l32_64e_top2_adaptation.me_fp16.bm_none.tps1024.samplealpha1.0.nlangs_32.bl1.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64.32k/checkpoint_59_96000.pt",
        "dict_path": "CC100_COMBINED_DICT_PATH",
        "extra_args": [
            "--batch-size",
            "1",
            "--is-moe",
        ],
        "model_overrides": {
            "world_size": 8,
            "langs": "en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE",
            "bpe": "sentencepiece",
            "sentencepiece_model": CC100_COMBINED_SPM,
        },
    },
}


def new_model_configs_by_checkpoint_paths(
    base_model_name: str,
    base_config: Dict[str, Any],
    checkpoint_id_to_path: Dict[int, str],
):
    """Generates new model configs based on checkpoint id and paths.
    Args:
        base_model_name (str): The name of the old setting config.
        base_config (Dict[str, Any]): The old model_config.
        checkpoint_id_to_path (Dict[int, str]): Id to Path maps.
    """
    new_model_configs = {}
    for checkpoint_id, file_path in checkpoint_id_to_path.items():
        new_model_name = f"{base_model_name}__step{checkpoint_id:08d}"
        new_config = copy.deepcopy(base_config)
        new_config["model_path"] = file_path

        new_model_configs[new_model_name] = new_config

    return new_model_configs


EXPANDED_MODEL_CONFIGS = {}  # These are populated in expand_model_configs


def expand_model_configs():
    """
    More models can be defined here by some programatic ways.
    """

    if len(EXPANDED_MODEL_CONFIGS) > 0:
        # already expanded
        return

    # Add 1.3B_gpt3_setting checkpoints
    curr_model_group = UNIDIR_LM_ROBERTA_DATA
    base_model_name = "1.3B_gpt3_setting"
    base_config = curr_model_group[base_model_name]

    base_path_early = "/checkpoint/myleott/2021-08-02/xlmg.1_3b.v2.fsdp.me_fp16.transformer_lm_gpt.nlay24.emb2048.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu375.dr0.1.atdr0.1.wd0.01.ms8.uf2.mu286102.s1.ngpu32/"
    checkpoint_id_to_file = {
        1000: os.path.join(base_path_early, "checkpoint_1_1000-shard0.pt"),
        2000: os.path.join(base_path_early, "checkpoint_1_2000-shard0.pt"),
        3000: os.path.join(base_path_early, "checkpoint_1_3000-shard0.pt"),
        4000: os.path.join(base_path_early, "checkpoint_1_4000-shard0.pt"),
        5000: os.path.join(base_path_early, "checkpoint_1_5000-shard0.pt"),
        6000: os.path.join(base_path_early, "checkpoint_1_6000-shard0.pt"),
        7000: os.path.join(base_path_early, "checkpoint_1_7000-shard0.pt"),
        8000: os.path.join(base_path_early, "checkpoint_1_8000-shard0.pt"),
        9000: os.path.join(base_path_early, "checkpoint_1_9000-shard0.pt"),
        10000: os.path.join(base_path_early, "checkpoint_1_10000-shard0.pt"),
        11000: os.path.join(base_path_early, "checkpoint_1_11000-shard0.pt"),
        12000: os.path.join(base_path_early, "checkpoint_1_12000-shard0.pt"),
        13000: os.path.join(base_path_early, "checkpoint_1_13000-shard0.pt"),
        14000: os.path.join(base_path_early, "checkpoint_1_14000-shard0.pt"),
        # 15000: os.path.join(base_path, "checkpoint_1_15000-shard0.pt"), # Available soon!
    }

    base_path = "/large_experiments/xlmg/models/dense/1.3B/few_shot.roberta+cc100.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/"
    checkpoint_id_to_file.update(
        {
            20000: os.path.join(base_path, "checkpoint_1_20000.pt"),
            50000: os.path.join(base_path, "checkpoint_1_50000.pt"),
            100000: os.path.join(base_path, "checkpoint_1_100000.pt"),
            150000: os.path.join(base_path, "checkpoint_2_150000.pt"),
            200000: os.path.join(base_path, "checkpoint_2_200000.pt"),
            250000: os.path.join(base_path, "checkpoint_3_250000.pt"),
        }
    )
    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)

    # Add 2.7B_gpt3_setting checkpoints
    curr_model_group = UNIDIR_LM_ROBERTA_DATA
    base_model_name = "2.7B_gpt3_setting"
    base_config = curr_model_group[base_model_name]
    base_path = "/large_experiments/xlmg/models/dense/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128/"
    checkpoint_id_to_file = {
        20000: os.path.join(base_path, "checkpoint_1_20000-shard0.pt"),
        50000: os.path.join(base_path, "checkpoint_1_50000-shard0.pt"),
        100000: os.path.join(base_path, "checkpoint_1_100000-shard0.pt"),
        150000: os.path.join(base_path, "checkpoint_2_150000-shard0.pt"),
        200000: os.path.join(base_path, "checkpoint_2_200000-shard0.pt"),
        250000: os.path.join(base_path, "checkpoint_3_250000-shard0.pt"),
    }
    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)

    # Add moe_52B checkpoints
    curr_model_group = UNIDIR_LM_ROBERTA_DATA
    base_model_name = "moe_52B"
    base_config = curr_model_group[base_model_name]
    base_path = "/large_experiments/xlmg/models/moe/52B/xlmg.52b.fp16.bm_none.tps2048.transformer_lm_gpt2_bigger.dl24.demb1024.dffn4096.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.sqrt_world_size.wu715.dr0.0.atdr0.0.wd0.01.ms2.uf1.mu572204.s1.ngpu128/"
    checkpoint_id_to_file = {
        105000: os.path.join(base_path, "checkpoint_1_105000_eval/checkpoint_eval.pt"),
        285000: os.path.join(base_path, "checkpoint_2_285000_eval/checkpoint_eval.pt"),
        370000: os.path.join(base_path, "checkpoint_2_370000_eval/checkpoint_eval.pt"),
        467500: os.path.join(base_path, "checkpoint_3_467500_eval/checkpoint_eval.pt"),
    }
    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)

    # Add moe_15B checkpoints
    curr_model_group = UNIDIR_LM_ROBERTA_DATA
    base_model_name = "moe_15B"
    base_config = curr_model_group[base_model_name]
    base_path = "/large_experiments/xlmg/models/moe/15B/xlmg.15b.fsdp.me_fp16.transformer_lm_gpt.nlay12.emb768.nexprt512.moe_w0.01.sqrt_world_size.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.0006.wu750.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu572204.s1.ngpu64/"
    checkpoint_id_to_file = {
        20000: os.path.join(base_path, "checkpoint_1_20000/checkpoint_1_20000.pt"),
        40000: os.path.join(base_path, "checkpoint_1_40000/checkpoint_1_40000.pt"),
        60000: os.path.join(base_path, "checkpoint_1_60000/checkpoint_1_60000.pt"),
        80000: os.path.join(base_path, "checkpoint_1_80000/checkpoint_1_80000.pt"),
        100000: os.path.join(base_path, "checkpoint_1_100000/checkpoint_1_100000.pt"),
        120000: os.path.join(base_path, "checkpoint_1_120000/checkpoint_1_120000.pt"),
        140000: os.path.join(base_path, "checkpoint_1_140000/checkpoint_1_140000.pt"),
        160000: os.path.join(base_path, "checkpoint_1_160000/checkpoint_1_160000.pt"),
        180000: os.path.join(base_path, "checkpoint_1_180000/checkpoint_1_180000.pt"),
        200000: os.path.join(base_path, "checkpoint_1_200000/checkpoint_1_200000.pt"),
    }
    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)

    # Add 6.7B_gpt3_setting checkpoints
    curr_model_group = UNIDIR_LM_ROBERTA_DATA
    base_model_name = "6.7B_gpt3_setting"
    base_config = curr_model_group["6.7B_gpt3_setting"]
    base_path = "/large_experiments/xlmg/models/dense/6.7B/xlmg_h2_2021.6_7b.fsdp.me_fp16.transformer_lm_gpt.nlay32.emb4096.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.00012.wu187.dr0.1.atdr0.1.wd0.01.ms8.uf1.mu143051.s1.ngpu128/"
    checkpoint_id_to_file = {
        10000: os.path.join(
            base_path, "checkpoint_1_10000/checkpoint_1_10000-shard0.pt"
        ),
        30000: os.path.join(
            base_path, "checkpoint_1_30000/checkpoint_1_30000-shard0.pt"
        ),
        50000: os.path.join(
            base_path, "checkpoint_1_50000/checkpoint_1_50000-shard0.pt"
        ),
        70000: os.path.join(
            base_path, "checkpoint_2_70000/checkpoint_2_70000-shard0.pt"
        ),
        90000: os.path.join(
            base_path, "checkpoint_2_90000/checkpoint_2_90000-shard0.pt"
        ),
        110000: os.path.join(
            base_path, "checkpoint_3_110000/checkpoint_3_110000-shard0.pt"
        ),
        130000: os.path.join(
            base_path, "checkpoint_3_130000/checkpoint_3_130000-shard0.pt"
        ),
        143050: os.path.join(base_path, "checkpoint_last/checkpoint_last-shard0.pt"),
    }

    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)

    # Add dense_7.5B_lang30_new_cc100_xl_unigram checkpoints
    curr_model_group = MULTI_LM_CC100_XL_SUPER_SHARD_DATA
    base_model_name = "dense_7.5B_lang30_new_cc100_xl_unigram"
    base_config = curr_model_group[
        "dense_7.5B_lang30_new_cc100_xl_unigram__step00065000"
    ]
    base_path = "/large_experiments/xlmg/models/multilingual/dense/7.5B/"
    checkpoint_id_to_file = {
        5000: os.path.join(base_path, "checkpoint_1_5000-shard0.pt"),
        10000: os.path.join(base_path, "checkpoint_1_10000-shard0.pt"),
        15000: os.path.join(base_path, "checkpoint_2_15000-shard0.pt"),
        20000: os.path.join(base_path, "checkpoint_2_20000-shard0.pt"),
        25000: os.path.join(base_path, "checkpoint_3_25000-shard0.pt"),
        30000: os.path.join(base_path, "checkpoint_3_30000-shard0.pt"),
        35000: os.path.join(base_path, "checkpoint_4_35000-shard0.pt"),
        50000: os.path.join(base_path, "checkpoint_5_50000-shard0.pt"),
        55000: os.path.join(base_path, "checkpoint_5_55000-shard0.pt"),
        60000: os.path.join(base_path, "checkpoint_6_60000-shard0.pt"),
        90000: os.path.join(base_path, "checkpoint_8_90000-shard0.pt"),
        100000: os.path.join(base_path, "checkpoint_9_100000-shard0.pt"),
        120000: os.path.join(base_path, "checkpoint_11_120000-shard0.pt"),
        150000: os.path.join(base_path, "checkpoint_13_150000-shard0.pt"),
        180000: os.path.join(base_path, "checkpoint_16_180000-shard0.pt"),
        210000: os.path.join(base_path, "checkpoint_19_210000-shard0.pt"),
        235000: os.path.join(base_path, "checkpoint_21_235000-shard0.pt"),
        238000: os.path.join(base_path, "checkpoint_last-shard0.pt"),
    }
    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)

    # Add moe_200B_lang30_new_cc100_xl_unigram checkpoints
    curr_model_group = MULTI_LM_CC100_XL_SUPER_SHARD_DATA
    base_model_name = "moe_200B_lang30_new_cc100_xl_unigram"
    base_config = curr_model_group["moe_200B_lang30_new_cc100_xl_unigram__step00048000"]
    base_path = "/large_experiments/xlmg/models/multilingual/moe/200B/200b.fsdp.zero2.me_fp16.transformer_lm_gpt2_big_wide.nlay48.emb2048.nexprt256.moe_w0.01.top1.sqrt.nlangs30.alpha1.0.bm_none.tps2048.stable.blockwise.adam8bit.fp16adam.b2_0.98.eps1e-08.cl1.0.lr0.00012.wu4000.dr0.1.atdr0.1.wd0.0.ms4.uf4.mu119209.s2.ngpu128/"
    checkpoint_id_to_file = {
        60000: os.path.join(
            base_path, "checkpoint_15_60000/checkpoint_15_60000_consolidated.pt"
        ),
        96000: os.path.join(
            base_path, "checkpoint_21_96000/checkpoint_21_96000-consolidated.pt"
        ),
    }
    checkpoint_configs = new_model_configs_by_checkpoint_paths(
        base_model_name, base_config, checkpoint_id_to_file
    )

    # TODO: move 200B multilingual MoE checkpoints and reorganize the directory
    EXPANDED_MODEL_CONFIGS.update(checkpoint_configs)


def get_model_config_groups():
    expand_model_configs()

    return {
        "HUGGINGFACE_API_DUMMY_MODELS": HUGGINGFACE_API_DUMMY_MODELS,
        "OPENAI_API_DUMMY_MODELS": OPENAI_API_DUMMY_MODELS,
        "DUMMY_MODELS": DUMMY_MODELS,
        "UNIDIR_LM_ROBERTA_DATA": UNIDIR_LM_ROBERTA_DATA,
        "FLAN_MODELS": FLAN_MODELS,
        "MULTI_LM_OLD_CC100_XL_DATA": MULTI_LM_OLD_CC100_XL_DATA,
        "MULTI_LM_CC100_DATA": MULTI_LM_CC100_DATA,
        "UNIDIR_LM_PILE_DATA": UNIDIR_LM_PILE_DATA,
        "MULTI_LM_CC100_COMBINED_ROBERTA": MULTI_LM_CC100_COMBINED_ROBERTA,
        "MULTI_LM_CC100_COMBINED_ROBERTA_ALPHA_SWEEP": MULTI_LM_CC100_COMBINED_ROBERTA_ALPHA_SWEEP,
        "MULTI_LM_CC100_XL_300_SHARD_DATA": MULTI_LM_CC100_XL_300_SHARD_DATA,
        "MULTI_LM_CC100_XL_SUPER_SHARD_DATA": MULTI_LM_CC100_XL_SUPER_SHARD_DATA,
        "MULTI_LM_CONTINUOUS_TRAINING_EXPERIMENTS": MULTI_LM_CONTINUOUS_TRAINING_EXPERIMENTS,
        "CC_100_XL_SPM_EXPERIMENTAL_MULTILINGUAL_RUN": CC_100_XL_SPM_EXPERIMENTAL_MULTILINGUAL_RUN,
        "EXPANDED_CONFIGS": EXPANDED_MODEL_CONFIGS,
    }


def get_model_configs():
    model_config_groups = get_model_config_groups()

    all_model_configs = {}

    for k, v in model_config_groups.items():
        all_model_configs.update(v)

    return all_model_configs


def get_model_names(re_filter="*"):
    model_names = get_model_configs().keys()

    if re_filter != "*":
        model_names = [mn for mn in model_names if re.match(re_filter, mn)]

    return model_names


def get_model_checkpoint_names(model_prefix, steps=None):
    return get_model_names(re.escape(model_prefix) + "__step*")


def check_model_paths():
    valid_names = []
    invalid_names = {}
    for c in get_model_config_groups().values():
        for k, v in c.items():
            if "model_path" not in v:
                invalid_names[k] = None
                continue
            parent = Path(v["model_path"]).parent
            has_subdir = parent.exists()
            if has_subdir:
                valid_names.append(k)
            else:
                invalid_names[k] = parent
    return valid_names, invalid_names


model_metadata = {
    "125M_gpt3_setting": {
        "Config names": "125M_gpt3_setting",
        "Model": "GPT-3 125M",
        "# params (B)": 0.125,
        "layers": 12,
        "hidden": 768,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 363560141,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 140,
        "Notes": "",
    },
    "openai_ada": {
        "Config names": "openai_ada, 355M_gpt3_setting",
        "Model": "GPT-3 355M",
        "# params (B)": 0.355,
        "layers": 24,
        "hidden": 1024,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 1058527642,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 408,
        "Notes": "",
    },
    "355M_gpt3_setting": {
        "Config names": "openai_ada, 355M_gpt3_setting",
        "Model": "GPT-3 355M",
        "# params (B)": 0.355,
        "layers": 24,
        "hidden": 1024,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 1058527642,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 408,
        "Notes": "",
    },
    "1.3B_gpt3_setting": {
        "Config names": "1.3B_gpt3_setting",
        "Model": "GPT-3 1.3B",
        "# params (B)": 1.3,
        "layers": 24,
        "hidden": 2048,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 3566606746,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 1376,
        "Notes": "",
    },
    "openai_babbage": {
        "Config names": "openai_babbage, 2.7B_gpt3_setting",
        "Model": "GPT-3 2.7B",
        "# params (B)": 2.7,
        "layers": 32,
        "hidden": 2560,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 7075504128,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 2730,
        "Notes": "",
    },
    "2.7B_gpt3_setting": {
        "Config names": "openai_babbage, 2.7B_gpt3_setting",
        "Model": "GPT-3 2.7B",
        "# params (B)": 2.7,
        "layers": 32,
        "hidden": 2560,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 7075504128,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 2730,
        "Notes": "",
    },
    "openai_curie": {
        "Config names": "openai_curie",
        "Model": "GPT-3 6.7B",
        "# params (B)": 6.7,
        "layers": 32,
        "hidden": 4096,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 17119012454,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 6605,
        "Notes": "",
    },
    "6.7B_gpt3_setting_1024ctx": {
        "Config names": "6.7B_gpt3_setting_1024ctx",
        "Model": "Our 6.7B",
        "# params (B)": 6.7,
        "layers": 32,
        "hidden": 4096,
        "seq len": 1024,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 16474767360,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 6356,
        "Notes": "",
    },
    "openai_davinci": {
        "Config names": "openai_davinci",
        "Model": "GPT-3 175B",
        "# params (B)": 175,
        "layers": 96,
        "hidden": 12288,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 0,
        "Extra MoE FLOPS per update": 0,
        "TFLOPS to train": 430173152870,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 165962,
        "Notes": "",
    },
    "moe_15B": {
        "Config names": "moe_15B",
        "Model": "MoE 15B",
        "# params (B)": 15,
        "layers": 12,
        "hidden": 768,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 512,
        "Extra MoE FLOPS per update": 67947725,
        "TFLOPS to train": 431507866,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 166,
        "Notes": "Matches 125M dense",
    },
    "moe_52B": {
        "Config names": "moe_52B",
        "Model": "MoE 52B",
        "# params (B)": 52,
        "layers": 24,
        "hidden": 1024,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 512,
        "Extra MoE FLOPS per update": 241591661,
        "TFLOPS to train": 1300118212,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 502,
        "Notes": "Matches 355M dense",
    },
    "moe_207B": {
        "Config names": "moe_207B",
        "Model": "MoE 207B",
        "# params (B)": 207,
        "layers": 24,
        "hidden": 2048,
        "seq len": 2048,
        "train tokens": "300.0 B",
        "# experts": 512,
        "Extra MoE FLOPS per update": 966366645,
        "TFLOPS to train": 4532969714,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 1749,
        "Notes": "Matches 1.3B dense",
    },
    "moe_523B": {
        "Config names": "moe_523B",
        "Model": "MoE 523B",
        "# params (B)": 523,
        "layers": 24,
        "hidden": 2304,
        "seq len": 1024,
        "train tokens": "302.0 B",
        "# experts": 1024,
        "Extra MoE FLOPS per update": 1231171548,
        "TFLOPS to train": 5407015280,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 2086,
        "Notes": "Doesn't match any dense model; has more experts",
    },
    "moe_1.1T": {
        "Config names": "moe_1.1T",
        "Model": "MoE 1.1T",
        "# params (B)": 1104,
        "layers": 32,
        "hidden": 4096,
        "seq len": 1024,
        "train tokens": "300.0 B",
        "# experts": 512,
        "Extra MoE FLOPS per update": 5153883385,
        "TFLOPS to train": 21628403428,
        "V100 TFLOPS": 30,
        "V100 GPU days to train": 8344,
        "Notes": "Matches 6.7B dense",
    },
}


if __name__ == "__main__":
    expand_model_configs()
    valid_names, invalid_names = check_model_paths()
    print(f"Supported models (valid): {valid_names}")
    print(f"Invalid: {invalid_names}")
    for valid_name in valid_names:
        print(valid_name)
