from fire import Fire
import torch
import re
from fairseq.file_io import load_and_pop_last_optimizer_state
from tqdm import tqdm
from typing import Dict


def subset(df, pat):
    return df[[x for x in df.columns if pat in x]]


def stitch_experts_to_dense(pth_prefix: str, lw_best_mask: Dict, to_dense=True) -> Dict:
    if to_dense:
        new_sd = {}
    else:
        from collections import defaultdict
        new_sd = defaultdict(dict)
    for layer, experts in tqdm(list(lw_best_mask.items())):
        if to_dense: assert len(experts) == 1
        for i, expert in enumerate(experts):
            expert_path = f'{pth_prefix}-rank-{expert}.pt'
            expert = load_and_pop_last_optimizer_state(expert_path)

            desired_param_prefix = f'decoder.layers.{layer}.'
            keep_params = {k: v for k, v in expert['model'].items() if k.startswith(desired_param_prefix)}
            assert keep_params, expert.keys()
            if to_dense:
                new_sd.update(keep_params)
            else:
                new_sd[i].update(keep_params)
    return new_sd


def remove_moe(x):
    return re.sub(r'moe_layer.experts.\d.', '', x)

def moe2dense(pth_prefix, save_path, fc_rank=0, random_linear=False, mask_path=None) -> None:
    if mask_path is None:  # just take rank 0
        expert_path = f'{pth_prefix}-rank-{fc_rank}.pt'
        expert = load_and_pop_last_optimizer_state(expert_path)
        fc_params = {remove_moe(k): v for k, v in expert['model'].items()}
        if random_linear:
            fc2 = {}
            for k,v in fc_params.items():
                if 'bias' in k:
                    fc2[k] = torch.zeros_like(v)
                elif 'weight' in k:
                    fc2[k] = torch.nn.Linear(v.shape[1], v.shape[0]).weight.data
            fc_params = fc2
    else:
        mask = torch.load(mask_path, map_location=torch.device("cpu"))
        fc_params = stitch_experts_to_dense(pth_prefix, mask, to_dense=True)
        fc_params = {remove_moe(k): v for k, v in fc_params.items()}

    full_st = load_and_pop_last_optimizer_state(pth_prefix + '-shared.pt')
    model = full_st['model']
    model.update(fc_params)
    model = {k: v for k, v in model.items() if 'wg.weight' not in k}
    full_st['model'] = model
    torch.save(full_st, save_path)


if __name__ == '__main__':
    Fire(moe2dense)


"""
ED=/large_experiments/xlmg/models/sshleifer/expert_drop/
python scripts/moe2dense.py  $ED/baseline.dl16.d1024.ngpu64/checkpoint_7_70000 $ED/baseline.dl16.d1024_dense_r0.pt
python scripts/moe2dense.py  $ED/continue_baseline2.dl32.d3072.ngpu64/checkpoint_7_65000 $ED/dl32.d3072_dense_r0.pt


"""

