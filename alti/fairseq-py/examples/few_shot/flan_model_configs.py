from examples.few_shot.utils import (
    PATH_TO_ROBERTA_DICT,
    dense_bpe_config,
    moe_bpe_config,
)

FLAN_MODELS = {
    "flan_minus_sentiment_2.7B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_sentiment.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.2.7B_gpt3_setting.chkact.ddpfs.bsz1.uf4.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm225.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_sentiment_6.7B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_sentiment.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.6.7B_gpt3_setting.chkact.ddpfs.bsz1.uf4.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm225.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_sentiment_13B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_sentiment.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.13B_gpt3_setting.chkact.ddpfs.bsz1.uf8.mu1875.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm112.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_qa_2.7B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_qa.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.2.7B_gpt3_setting.chkact.ddpfs.bsz1.uf4.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm225.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_qa_6.7B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_qa.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.6.7B_gpt3_setting.chkact.ddpfs.bsz1.uf4.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm225.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_qa_13B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_qa.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.13B_gpt3_setting.chkact.ddpfs.bsz1.uf8.mu1875.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm112.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_qa_13B_v2": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_qa.nonews..ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.13B_gpt3_setting.chkact.ddpfs.bsz4.uf2.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr1e-05.warm112.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_nli_para_2.7B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_nli_para.nonews.ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.2.7B_gpt3_setting.chkact.ddpfs.bsz1.uf4.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm225.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_nli_para_6.7B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_nli_para.nonews.ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.6.7B_gpt3_setting.chkact.ddpfs.bsz1.uf4.mu3750.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm225.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
    "flan_minus_nli_para_13B": dense_bpe_config(
        "/checkpoint/sviyer/gshard_models/flan_minus_nli_para.nonews.ft_whole.pr0.cheat.eps_3000.sbm_eos.tok_768.13B_gpt3_setting.chkact.ddpfs.bsz1.uf8.mu1875.dr0.1.atdr0.1.actdr0.0.adam..lr0.0001.warm112.me_fp16.ngpu128/checkpoint_last-shard0.pt"
    ),
}
