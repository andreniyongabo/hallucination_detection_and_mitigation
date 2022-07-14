#!/usr/bin/env python
"""
Example usage:

    PYTHONPATH=. ./fb_sweep/xlmg/sweep_xlmg_multilingual_lm.py \
            --num-trials 1 --num-gpus 8 --num-nodes 1 \
            --model-size 150M_multilm_h2_2021 \
            --langs-key test3 \
            --prefix test3 \
            --benchmark \
            --partition learnaccel

This sweep script takes some additional optional arguments. See add_extra_options_func

For 8bit adam, needs to install bitsandbytes:
    pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda110 -U
See https://gist.github.com/sshleifer/e02c30a66a94126aea6baf77df926e84 for full details.

for more details.
"""

import os
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


def add_extra_options_func(parser):
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use 3 languages and only train for 100 steps (for benchmarking)",
    )
    parser.add_argument(
        "--model-size", help="model configuration, see get_grid for available options"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="tokens_per_sample")
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--debug-train-on-small-subset",
        action="store_true",
        help="only load a single shard of data from one datasource (OpenWebText), "
        "which reduces startup time and is useful for debugging",
    )
    parser.add_argument(
        "--langs-key",
        help="languages configuration, see get_grid for available options",
    )


def infer_data_path_(args):
    # TODO: update to full shards
    num_shards = 100
    if os.path.exists("/nfs2/"):  # Azure
        data_bin = "TODO"
        args.snapshot_code = False  # containers don"t support snapshot_code
        args.data = ":".join([f"{data_bin}/shard{i}" for i in range(num_shards)])
    elif os.path.exists("/fsx"):  # AWS
        # TODO: update to full shards
        data_bin = "/fsx/myleott/data/test_sample_cc100_xl_unigram"
        num_shards = 1
        args.data = ":".join(
            [f"{data_bin}/sample-shard-{i}" for i in range(num_shards)]
        )
    else:
        # TODO: update to full shards
        data_bin = "/large_experiments/moe/cc100_xl_roberta/final_bin"
        args.snapshot_code = False if args.local else True
        args.data = ":".join([f"{data_bin}/shard{i}" for i in range(num_shards)])


def get_base_model_config(layers, model_dim, heads, ffn_dim=None):
    if ffn_dim is None:
        ffn_dim = 4 * model_dim
    return [
        hyperparam(
            "--arch", "transformer_lm_gpt2_big_wide", save_dir_key=lambda val: val
        ),
        hyperparam("--activation-fn", "gelu"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--decoder-layers", layers, save_dir_key=lambda val: f"nlay{val}"),
        hyperparam(
            "--decoder-embed-dim", model_dim, save_dir_key=lambda val: f"emb{val}"
        ),
        hyperparam("--decoder-ffn-embed-dim", ffn_dim),
        hyperparam("--decoder-attention-heads", heads),
    ]


def add_moe_config_(model_config, expert_count):
    model_config.extend(
        [
            hyperparam(
                "--moe-expert-count",
                expert_count,
                save_dir_key=lambda val: f"nexprt{val}",
            ),
            hyperparam("--moe-freq", 2),  # MOE on every other layer
            hyperparam("--criterion", "moe_cross_entropy"),
            hyperparam(
                "--moe-gate-loss-wt", [0.01], save_dir_key=lambda val: f"moe_w{val}"
            ),
            hyperparam("--moe-gating-use-fp32"),
            hyperparam("--moe-gate-loss-combine-method", "sum"),
            hyperparam("--moe-second-expert-policy", "all"),
            # hyperparam("--moe-normalize-gate-prob-before-dropping", save_dir_key=lambda val: "norm_b"),
            hyperparam(
                "--moe-normalize-expert-grad",
                "sqrt_world_size",
                save_dir_key=lambda val: val,
            ),
            hyperparam("--pad-to-fixed-length"),
            hyperparam(
                "--moe-eval-capacity-token-fraction", -1.0
            ),  # use same capacity during valid and train
            hyperparam(
                "--max-sentences-valid", 1
            ),  # not strictly necessary, but safer to avoid OOM
            hyperparam("--num-workers-valid", 0),  # this can avoid hangs in some cases
        ]
    )


def get_grid(args):
    num_gpus = args.num_gpus * args.num_nodes
    training_tokens = int(500e9)  # GPT-3 300B

    # Set this to 0 on AWS to avoid segfaults
    num_dataloading_workers = 2 if not os.path.exists("/fsx") else 0
    train_subset = "train"

    # TODO the original dense training runs in H1 2021 used a single validation
    # set coming from CC-News. If you need to have comparable valid_ppl to those
    # runs, then set this to False. Otherwise True is preferred, since it will
    # aggregate all of the valid sets for CC-News, Books, Wikipedia, etc.
    combine_valid_sets = True

    # Infer data path if not given
    if args.data is None:
        infer_data_path_(args)
    assert os.path.exists(
        args.data.split(":")[0]
    ), f"Could not find data path: {args.data}"

    # Sampling language L with p(|L|)^{1/sampling_alpha}
    # sampling_alpha = 1: proportional sampling
    # sampling_alpha < 1: upsampling low resource
    # sampling_alpha > 1: upsampling high resource
    sampling_alpha = 1.0
    langs_dict = {
        "test3": "en_XX,hi_IN,ss_SZ",
        "eval28": (
            "en_XX,et_EE,ht_HT,id_ID,it_IT,qu_PE,sw_KE,ta_IN,th_TH,tr_TR,vi_VN,zh_CN,ar_AR,bg_BG,de_DE,es_XX,"
            "fr_XX,hr_HR,hu_HU,lt_LT,mk_MK,pl_PL,pt_XX,sq_AL,sr_RS,el_GR,ru_RU,hi_IN"
        ),
        "all134": (
            "en_XX,es_XX,de_DE,fr_XX,ja_XX,zh_CN,ru_RU,it_IT,pt_XX,el_GR,ro_RO,uk_UA,hu_HU,nl_XX,id_ID,pl_PL,"
            "da_DK,no_XX,fi_FI,hr_HR,ko_KR,tr_TR,th_TH,vi_VN,ar_AR,ms_MY,fa_IR,bg_BG,sv_SE,zh_TW,cs_CZ,he_IL,"
            "sk_SK,ca_ES,lt_LT,hi_IN,hi_IN_rom,sl_SI,tl_XX,et_EE,lv_LV,sq_AL,bn_IN,ta_IN,sr_RS,az_AZ,ar_AR_rom,"
            "ur_PK,kk_KZ,is_IS,sw_KE,ka_GE,hy_AM,mk_MK,af_ZA,jv_ID,ml_IN,be_BY,la_VA,mn_MN,ne_NP,te_IN,bs_BA,"
            "mr_IN,ur_PK_rom,cy_GB,so_SO,ps_AF,si_LK,kn_IN,bn_IN_rom,km_KH,gu_IN,uz_UZ,cb_IQ,ha_NG,gl_ES,su_ID,"
            "pa_IN,mg_MG,cx_PH,eu_ES,ht_HT,sa_IN,am_ET,lo_LA,my_MM,or_IN,ta_IN_rom,br_FR,eo_EO,ky_KG,te_IN_rom,"
            "ga_IE,yo_NG,az_IR,my_MM_zaw,zu_ZA,gd_GB,ku_TR,ci_IT,qh_PH,xh_ZA,ig_NG,om_KE,as_IN,ti_ET,wo_SN,tn_BW,"
            "ns_ZA,fy_NL,ff_NG,lg_UG,qj_ML,sd_PK,ln_CD,qf_CM,qd_MM,gn_PY,qy_KE,bm_ML,q3_CV,qx_KE,qm_AO,iu_CA,"
            "qu_PE,kg_AO,ql_ML_rom,ss_SZ,qw_KE,qi_PH_rom,qa_MM,qq_KE,qp_AO,"
        ),
    }

    # Model configuration based on size
    M = 1024 * 1024
    if args.model_size == "150M_multilm_h2_2021":
        # assert num_gpus >= 8
        model_config = get_base_model_config(layers=6, model_dim=512, heads=16)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.5e-3
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.0

    elif args.model_size == "564M_multilm_h2_2021":
        # assert num_gpus >= 32
        model_config = get_base_model_config(layers=24, model_dim=1024, heads=16)
        batch_size_tokens = int(2 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.5e-3
        warmup_tokens = int(8000 * M)
        dropout = 0.1
        weight_decay = 0.0
    elif args.model_size == "1.7B_multilm_h2_2021":
        assert num_gpus >= 32
        model_config = get_base_model_config(layers=24, model_dim=2048, heads=16)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.5e-3
        warmup_tokens = int(8000 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "2.9B_multilm_h2_2021":
        # assert num_gpus >= 128
        model_config = get_base_model_config(layers=48, model_dim=2048, heads=16)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.5e-3
        warmup_tokens = int(8000 * M)
        dropout = 0.1
        weight_decay = 0.0
    elif args.model_size == "6.9B_multilm_h2_2021":
        # assert num_gpus >= 128
        model_config = get_base_model_config(layers=54, model_dim=3072, heads=24)
        batch_size_tokens = int(4.0 * M)
        max_batch_size_per_gpu = 2
        learning_rate = 1.5e-3
        warmup_tokens = int(16000 * M)
        dropout = 0.1
        weight_decay = 0.0
    else:
        raise ValueError(f"Unknown --model-size argument: {args.model_size}")

    # Batch size logic
    batch_size_seqs = batch_size_tokens // args.seq_len
    batch_size_per_gpu = min(max_batch_size_per_gpu, batch_size_seqs // num_gpus)
    update_freq = batch_size_seqs // (batch_size_per_gpu * num_gpus)
    assert (
        batch_size_tokens == update_freq * batch_size_per_gpu * num_gpus * args.seq_len
    )

    max_update = training_tokens // batch_size_tokens
    warmup_updates = warmup_tokens // batch_size_tokens

    log_interval = 1 if args.local else 100

    task_config = [
        hyperparam("--task", "multilingual_language_modeling"),
        hyperparam(
            "--langs",
            langs_dict[args.langs_key],
            save_dir_key=lambda val: f"nlangs{len(val.split(','))}",
        ),
        hyperparam(
            "--multilang-sampling-alpha",
            sampling_alpha,
            save_dir_key=lambda val: f"alpha{val}",
        ),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam(
            "--tokens-per-sample", args.seq_len, save_dir_key=lambda val: f"tps{val}"
        ),
    ]
    if args.benchmark:
        # Overrides for speed benchmarking
        args.langs_key = "test3"
        batch_size_per_gpu = max_batch_size_per_gpu
        update_freq = 1
        max_update = 100
        warmup_updates = 50
        log_interval = 1

    grid = []
    if args.restore_file:
        grid += [
            hyperparam("--restore-file", args.restore_file),
            hyperparam("--reset-dataloader"),
            hyperparam("--reset-lr-scheduler"),
            hyperparam("--reset-meters"),
            hyperparam("--reset-optimizer"),
        ]
    grid += [
        hyperparam("--tensorboard-logdir", ""),
        hyperparam("--train-subset", train_subset),
        hyperparam("--num-workers", num_dataloading_workers),
        # hyperparam("--save-async"),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", 1000),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        # hyperparam("--keep-interval-updates", 1),  # only keep the most recent checkpoint
        # hyperparam("--no-save-optimizer-state-on-training-finished"),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda val: "fsdp"),
        hyperparam("--checkpoint-activations"),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 4),
        hyperparam("--threshold-loss-scale", 0.25),
    ]
    grid += model_config
    grid += task_config

    if args.model_size == "7.2B_multilm_h2_2021":
        grid += [
            hyperparam(
                "--use-stable-embedding", save_dir_key=lambda x: "stable" if x else ""
            ),
            hyperparam("--block-wise", save_dir_key=lambda x: "blockwise" if x else ""),
            hyperparam("--no-scale-embedding"),
            hyperparam("--optimizer", "adam8bit", save_dir_key=lambda val: val),
        ]
    else:
        grid += [
            hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        ]

    grid += [
        hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam"),
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            "(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam(
            "--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"
        ),  # GPT-3 used --clip-norm=1.0
        hyperparam("--clip-norm", 1.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", learning_rate, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--dropout", dropout, save_dir_key=lambda val: f"dr{val}"),
        hyperparam(
            "--attention-dropout", dropout, save_dir_key=lambda val: f"atdr{val}"
        ),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
        hyperparam(
            "--batch-size", batch_size_per_gpu, save_dir_key=lambda val: f"ms{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", log_interval),
    ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
