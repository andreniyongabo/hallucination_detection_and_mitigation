import shutil

from fire import Fire
from pathlib import Path
from tqdm import tqdm


def copy_shard_files_to_checkpoint_last(
    save_dir, pattern="checkpoint_2_386000", dest_dir=None
):
    if pattern.endswith("_"):
        pattern = pattern[:-1]
    assert not pattern.endswith(".pt"), pattern
    paths = list(Path(save_dir).glob(f"{pattern}*.pt"))
    assert paths, f"no paths matching {save_dir}/{pattern}*.pt"
    for p in tqdm(paths):
        new_path = str(p).replace(pattern, "checkpoint_last")
        if dest_dir is not None:
            new_path = new_path.replace(str(save_dir), dest_dir)
        shutil.copyfile(p, new_path)


if __name__ == "__main__":
    Fire(copy_shard_files_to_checkpoint_last)
