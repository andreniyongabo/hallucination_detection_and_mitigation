import os
import tempfile
import unittest

try:
    from examples.few_shot.gpt3_eval import (
        get_argparser,
        run_evaluations_from_model_name,
    )
except FileNotFoundError:
    raise unittest.SkipTest(
        "Unable to import tasks module. Skipping all tests in test_few_shot.py"
    )

import socket
from typing import Dict, List

import torch

from examples.few_shot.model_configs import check_model_paths
from fairseq.utils import assert_equal

HOSTNAME = socket.gethostname()
IS_FAIR = os.path.exists("/private/home/sshleifer")


def _run_task(
    model_name, task="copa", nshot=0, num_trials=1, extra_args=None
) -> List[Dict]:
    with tempfile.TemporaryDirectory() as results_dir:
        parser = get_argparser()
        input_args = [
            "--model-name",
            model_name,
            "--tasks",
            task,
            "--nshot",
            str(nshot),
            "--num-trials",
            str(num_trials),
            "--batch-size",
            "2",  # Does this do anything?
            "--results-dir",
            str(results_dir),
        ]
        if extra_args is not None:
            input_args += extra_args
        args, _ = parser.parse_known_args(input_args)
        args.train_sep = args.train_sep.replace(
            "\\n", "\n"
        )  # Newlines are escaped by argparse
        results = run_evaluations_from_model_name(**vars(args))
    scores = {
        k: v["mean"]
        for k, v in results[0].items()
        if isinstance(v, dict) and "mean" in v
    }
    return results, scores


@unittest.skipUnless(IS_FAIR, "FAIR Cluster only")
def test_all_models_have_good_paths():
    _, bad_paths = check_model_paths()
    if bad_paths:
        raise FileNotFoundError(bad_paths)


@unittest.skipUnless(IS_FAIR)
@unittest.skipIf(not torch.cuda.is_available(), "test requires CUDA")
@unittest.skipIf(
    "devfair" in HOSTNAME,
    "These tests dont work on devfair. Command: https://tinyurl.com/zud42up5 ",
)
class TestZeroShot(unittest.TestCase):
    def test_tiny_moe(self):
        results, scores = _run_task("tiny_moe", extra_args=["--n", "8"])

    def test_125M_gpt3_setting(self):
        results, scores = _run_task(
            "125M_gpt3_setting", extra_args=["--n", "8", "--max-tokens", "2048"]
        )
        assert_equal(scores["accuracy"], 62.5, msg=f"scores:{scores}")
        assert_equal(round(scores["ppl_selected_candidate"], 1), 27.5)
        # TODO(SS): This is 27.49 on some v100 and 27.51 on other v100

    def test_moe_multilingual(self):
        # This one takes 1 min with CUDA_VISIBLE_DEVICES=0,1
        results, scores = _run_task(
            "tiny_multi_moe",
            task="copa",
            extra_args=["--n", "8", "--cap", "1.", "--max-tokens", "2048"],
        )
        assert_equal(scores["accuracy"], 62.5, msg=f"scores:{scores}")


if __name__ == "__main__":
    unittest.main()
