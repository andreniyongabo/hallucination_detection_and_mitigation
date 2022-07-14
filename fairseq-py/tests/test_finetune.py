import tempfile
import unittest

import torch

import fairseq.distributed.utils as distributed_utils
from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_cli import train


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestFinetune(unittest.TestCase):
    def test_finetune(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp:
            train_parser = options.get_training_parser()
            args = [
                "--restore-file=/large_experiments/xlmg/models/dense/125M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu715.dr0.1.atdr0.1.wd0.01.ms4.uf2.mu572204.s1.ngpu32/checkpoint_best.pt",
                "--reset-optimizer",
                "--reset-dataloader",
                "--arch=transformer_lm",
                "--model-name=125M_gpt3_setting",
                "--user-dir=examples/few_shot/finetune",
                "--task=prompt_tuning",
                "--criterion=prompt_tuning",
                "--downstream-task=boolq",
                "--finetune-model-weights",
                "--sample-break-mode=eos",
                "--optimizer=adam",
                "--batch-size=1",
                "--max-update=1",
                "--no-save",
                "--log-file={}".format(temp.name),
            ]
            train_args = options.parse_args_and_arch(train_parser, args)
            cfg = convert_namespace_to_omegaconf(train_args)
            distributed_utils.call_main(cfg, train.main)

            lines = temp.readlines()
            assert lines[-1].startswith("done training")
            assert lines[-2].startswith("epoch 001")


if __name__ == "__main__":
    unittest.main()
