import os

from examples.few_shot.models import MULTI_LM_OLD_CC100_XL_DATA, MULTI_LM_CC100_DATA, MULTI_LM_CC100_COMBINED_ROBERTA, UNIDIR_LM_ROBERTA_DATA

from fairseq_cli.eval_lm import main
from fairseq import options, distributed_utils
from fire import Fire
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from pathlib import Path

XL_DIR = '/private/home/sshleifer/ThePile/spm_cc100_xl'
CC100_DIR = '/private/home/sshleifer/ThePile/spm_cc'
EN_DIR = "/private/home/myleott/data/data-bin/ThePile"

def eval_pile(mname, subset):
    output_name = f'{mname}.results/ThePile.{subset}.test_ppl.json'
    Path(output_name).parent.mkdir(exist_ok=True)
    if mname in MULTI_LM_OLD_CC100_XL_DATA:
        model = MULTI_LM_OLD_CC100_XL_DATA[mname]
        pile_dir = XL_DIR
    elif mname in MULTI_LM_CC100_COMBINED_ROBERTA:
        model = MULTI_LM_CC100_COMBINED_ROBERTA[mname]
        pile_dir = XL_DIR

    elif mname in MULTI_LM_CC100_DATA:
        model = MULTI_LM_CC100_DATA[mname]
        pile_dir = CC100_DIR
    elif mname in UNIDIR_LM_ROBERTA_DATA:
        model = UNIDIR_LM_ROBERTA_DATA[mname]
        pile_dir = EN_DIR
    else:
        raise KeyError(f'{mname} not found in supported subset of model configs. You probably need to update eval_pile.py ')

    data_dir = os.path.join(pile_dir, subset)
    overrides = model['model_overrides']
    eval_lm_param = [
                        data_dir,
                        "--path",
                        model['model_path'],
                        "--tokens-per-sample", '1024',
                        "--sp", output_name,
                        "--gen-subset", "test",
                        "--fp16",
                        "--model-overrides", str(overrides),
                    ] + model.get('extra_args', [])
    print(eval_lm_param)

    eval_lm_parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(eval_lm_parser, eval_lm_param)
    cfg = convert_namespace_to_omegaconf(args)
    # cfg.common_eval.model_overrides = overrides
    distributed_utils.call_main(cfg, main)


if __name__ == '__main__':
    Fire(eval_pile)

"""Usage:
python examples/few_shot/eval_pile.py 6.7B_gpt3_setting_1024ctx Enron_Emails

"""
