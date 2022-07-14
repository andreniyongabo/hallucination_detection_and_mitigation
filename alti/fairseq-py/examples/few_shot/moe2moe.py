from pathlib import Path
from moe2dense import stitch_experts_to_dense
from fairseq.file_io import load_and_pop_last_optimizer_state

from fire import Fire
import torch
from tqdm import tqdm
import re

def moe2moe(pth_prefix: str, save_dir: str, mask_path: str, save_prefix='start',
            copy_routing=False):
    if pth_prefix.endswith('.pt'):
        pth_prefix = pth_prefix[:-3]
    Path(save_dir).mkdir(exist_ok=True)
    mask = torch.load(mask_path, map_location=torch.device("cpu"))

    n_expert = len(list(mask.values())[0])  # Must be same for each layer.
    print(f'stitching: {n_expert} experts')
    fc_params = stitch_experts_to_dense(pth_prefix, mask, to_dense=False)
    dummy_expert = load_and_pop_last_optimizer_state(f'{pth_prefix}-rank-0.pt')
    for eid, expert in tqdm(fc_params.items(), desc='saving'):
        save_path = f'{save_dir}/{save_prefix}-rank-{eid}.pt'
        dummy_expert['model'] = expert
        torch.save(dummy_expert, save_path)
    shared_path = f'{save_dir}/{save_prefix}-shared.pt'
    shared = load_and_pop_last_optimizer_state(f'{pth_prefix}-shared.pt')
    # TODO(SS): it might help a bit to not reinitialize routing parameters. Could slice them using the mask.
    for k, model_param in shared['model'].items():
        if 'moe_layer.gate.wg.weight' in k:
            if copy_routing:
                lnum = int(re.match('decoder.layers.(\d*).moe_layer.gate.wg.weight', k).groups()[0])
                new_param = model_param[mask[lnum]]
                shared['model'][k] = new_param
            else:
                shared['model'][k] = torch.nn.Linear(model_param.shape[1], 32).weight.data
    torch.save(shared, shared_path)

if __name__ == '__main__':
    Fire(moe2moe)


"""
# Usage:
ED=/large_experiments/xlmg/models/sshleifer/expert_drop/
python moe2moe.py $ED/m2m.dl12.d2048.moe_w0.01.ngpu32/checkpoint_8_40000 \
    --save-dir m2m.dl12.d2048_best_24 \
    --mask-path /private/home/sshleifer/fairseq-py/m2m32_masks/good_mask24.pt

"""
