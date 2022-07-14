import glob
import os
from pathlib import Path
import re
import shutil
import json
import logging

def get_checkpoint_ids_from_text(checkpoint_ranges_specification: str):
    """ Get a list of checkpoint ids from text filter of ranges or checkpoint ids specified as:
        example: "0-10k:1k,10k-20k:5k,23000"
        where each comma-separated item is "{start}-{end}:{step}" or "{checkpoint_id}"
            
    Args:
        checkpoint_ranges_specification (str): Formatted ranges and steps like "0-10k:1k,10k-20k:5k,23000"

    Raises:
        ValueError: If some of the ranges "{start}-{end}:{step}" can not be parsed 

    Returns:
        List[int]: List of checkpoint ids
    """

    checkpoint_ranges_specification = checkpoint_ranges_specification.replace("k", "000")
    ids = []
    for range_with_step in checkpoint_ranges_specification.split(","):
        range_with_step = range_with_step.strip()  # "0-10000:1000" or "1000"
        if range_with_step.isdigit():
            id = int(range_with_step)
            ids.append(id)
            continue

        range_step_parse_re = re.search("([\d]+)-([\d]+):([\d]+)", range_with_step)
        if range_step_parse_re:
            range_start = int(range_step_parse_re.group(1))
            range_end = int(range_step_parse_re.group(2))
            step = int(range_step_parse_re.group(3))
        else:
            raise ValueError(f"`{range_with_step}` must be in the format `start-end:step` like `0-10k:1k` or `0-10000:1000`")
        
        for id in range(range_start, range_end, step):
            if id == 0:
                continue
            ids.append(id)
    
    return ids


def get_checkpoint_id_from_filename(checkpoint_file_path):
    """Extracts the checkpoint id from checkpoint filename.

    Args:
        checkpoint_basename (str): File basename

    Returns:
        Optional[int]: Checkpoint id. None is returned when we can not get id from the checkpoint file path. 
    """
    checkpoint_basename = os.path.basename(checkpoint_file_path)
    checkpoint_id_search = re.search('checkpoint.*_([\d]+)-shard.*', checkpoint_basename, re.IGNORECASE)
    if checkpoint_id_search:
        checkpoint_id_str = checkpoint_id_search.group(1)
        return int(checkpoint_id_str)
    else:
        return None


def delete_checkpoint_with_shards(checkpoint_path):
    """Deletes a checkpoint and all of its shards!

    Args:
        checkpoint_path ([type]): Full checkpoint path
    """
    if not os.path.exists(checkpoint_path):
        logging.warning("Checkpoint path {checkpoint_path} does not exist!")
        return
    
    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint_base_name = os.path.basename(checkpoint_path)
    checkpoint_parent_dir = checkpoints_parent_dir = Path(checkpoint_path).parent


    # Delete all shards
    checkpoint_shards = glob.glob(checkpoint_path.replace("-shard0.pt", "-shard*.pt"))
    for checkpoint_shard_file in checkpoint_shards:
        os.remove(checkpoint_shard_file)

    # When checkpoints are in a sub-directory with the same name prefix, we delete the full directory.
    if os.path.basename(checkpoint_parent_dir) == checkpoint_base_name.replace("-shard0.pt", ""):
        shutil.rmtree(checkpoint_parent_dir)
    

def read_json_config_file_and_populate_env_vars(file_path):
    res_json = None
    with open(file_path) as f:
        config_text = f.read()
        if "{AZURE_AUTH_" in config_text:
            azure_auth_settings = {k: os.getenv(k) for k in [
                                    "AZURE_AUTH_SP",
                                    "AZURE_AUTH_SE",
                                    "AZURE_AUTH_ST",
                                    "AZURE_AUTH_SIG",
                                  ]}
            if any([v is None for k, v in azure_auth_settings.items()]):
                raise EnvironmentError("Please, make sure that you have the Azure auth settings set in your environment variables:\n"
                                "export AZURE_AUTH_SP=VALUE\n"
                                "export AZURE_AUTH_SE=VALUE\n"
                                "export AZURE_AUTH_ST=VALUE\n"
                                "export AZURE_AUTH_SIG=VALUE\n"
                                )

            config_text = config_text.replace("{AZURE_AUTH_SP}", azure_auth_settings["AZURE_AUTH_SP"])
            config_text = config_text.replace("{AZURE_AUTH_SE}", azure_auth_settings["AZURE_AUTH_SE"])
            config_text = config_text.replace("{AZURE_AUTH_ST}", azure_auth_settings["AZURE_AUTH_ST"])
            config_text = config_text.replace("{AZURE_AUTH_SIG}", azure_auth_settings["AZURE_AUTH_SIG"])
        
        res_json = json.loads(config_text)
    return res_json


if __name__ == "__main__":
    test_checkpoint_path = "/large_experiments/xlmg/models/intermediate_eval_checkpoints/gptz.exp14.fsdp.me_fp16.zero2.transformer_lm_gpt.nlay24.emb2048.bm_none.tps2048.vanilla.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu375.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoint_3_15000/checkpoint_3_15000-shard0.pt"
    delete_checkpoint_with_shards(test_checkpoint_path)