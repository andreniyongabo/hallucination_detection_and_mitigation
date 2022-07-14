import json
import tempfile
import unittest

import torch

from examples.few_shot.gpt3_eval import get_argparser, run_evaluations_from_model_name


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestEval(unittest.TestCase):
    def test_eval(self):
        with tempfile.TemporaryDirectory() as tempdir:
            parser = get_argparser()
            args = parser.parse_args(
                [
                    "--model-name=125M_gpt3_setting",
                    "--tasks=copa",
                    "--nb-few-shot-samples-values=0",
                    "--results-dir={}".format(tempdir),
                ]
            )
            run_evaluations_from_model_name(**vars(args))

            with open(
                "{}/task.copa_tmp.copa_train.None.None_val.None.None_eval.val.en_calib.None_fs0_results.json".format(
                    tempdir
                )
            ) as f:
                results = json.load(f)

            assert results["model_name"] == "125M_gpt3_setting"
            assert results["task"] == "copa"


if __name__ == "__main__":
    unittest.main()
