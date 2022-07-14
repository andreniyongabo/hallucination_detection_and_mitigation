import torch
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path", type=str, metavar="PATH", help="model path without suffixes"
)
parser.add_argument(
    "--world-size", type=int, metavar="D", help="world size used for model training"
)


def convert_to_noc10d_format(model_path, world_size):
    model_dir = os.path.dirname(model_path)
    output_model_dir = model_dir + "_converted"
    os.makedirs(output_model_dir, exist_ok=True)
    fname_prefix = os.path.basename(model_path).split(".")[0]
    shared_sd = {}
    for i in range(world_size):
        fname = f"{fname_prefix}-rank-{i}-shard{i}.pt"
        print(f"reading {fname}")
        path = os.path.join(model_dir, fname)
        state = torch.load(path)
        shared_sd.update({k: v for k, v in state["model"].items() if "expert" not in k})
        expert_sd = {k: v for k, v in state["model"].items() if "expert" in k}
        state["model"] = expert_sd
        state["last_optimizer_state"] = {}

        ofname = f"{fname_prefix}-rank-{i}.pt"
        opath = os.path.join(output_model_dir, ofname)
        torch.save(state, opath)
    state["model"] = shared_sd
    ofname = f"{fname_prefix}-shared.pt"
    opath = os.path.join(output_model_dir, ofname)
    torch.save(state, opath)
    print("Done")


def cli_main():
    args = parser.parse_args()
    convert_to_noc10d_format(args.model_path, args.world_size)


if __name__ == "__main__":
    cli_main()
