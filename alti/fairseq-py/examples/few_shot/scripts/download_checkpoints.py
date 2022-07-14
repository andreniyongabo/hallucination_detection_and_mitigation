import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Union
import json
import logging
from examples.few_shot.scripts.checkpoint_helpers import get_checkpoint_ids_from_text, read_json_config_file_and_populate_env_vars


"""
Example usage
PYTHONPATH=. python examples/few_shot/scripts/download_checkpoints.py \
    -o /large_experiments/xlmg/models/intermediate_eval_checkpoints_debug/debug_download/ \
    --config-file examples/few_shot/scripts/experiments/configs/1.3B_gpt3_setting_kitchen_sink_14.json \
    --last-checkpoint \
    --checkpoint-id-filter 0-100k:10000 \
    
"""

def build_azure_blob_path(blob_model_path:str, azure_url:str, azure_url_params: Dict[str, any], read_only_sub_dir=False):
    """This builds up the azure blob path needed for downloading objects.

    Args:
        blob_model_path ([type]): [description]
        azure_url ([type]): This is the base Azure url.
        azure_url_params ([type]): Params used for url creation, authentication, etc.
        read_only_sub_dir (bool, optional): [description]. Defaults to False.

    Returns:
        str: Full azure url
    """
    if read_only_sub_dir:
        full_url = os.path.join(azure_url, blob_model_path, "*?")+ "&".join([ f"{k}={v}" for k,v in azure_url_params.items()])
    else: 
        full_url = os.path.join(azure_url, blob_model_path, "?")+ "&".join([ f"{k}={v}" for k,v in azure_url_params.items()])
    return full_url


def download_checkpoints_from_azure_blob(config_file_or_json: Union[str, Dict[str, any]], 
                                         output_path: str, 
                                         checkpoint_ids: List[int], 
                                         silent=False, 
                                         append_model_name_to_out_path=False
    ):
    """Downloads missing checkpoints from azure blob

    Args:
        config_file (str): This is the intermediate eval config file or config json. 
        output_path (str): The directory where the checkpoints will be stored.
        checkpoint_ids (List[int]): List of ids to download. If an item in the list is -1, it is replaced with the last checkpoint.  
        silent (bool, optional): Do not output debug information. Defaults to False.
        append_model_name_to_out_path (bool): Appends the model name to the output path. Defaults to False.
    """
    
    if isinstance(config_file_or_json, str):
        config_file = config_file_or_json
        config = read_json_config_file_and_populate_env_vars(config_file)
        config = config["azure_copy_setting"]
    else:
        config = config_file_or_json["azure_copy_setting"]

    input_path = config["blob_model_path"]
    model_name = os.path.basename(input_path)
    if append_model_name_to_out_path:
        full_model_path = os.path.join(output_path, model_name)
    else:
        full_model_path = output_path

    # List all the checkpoints from the input_path
    azure_path = build_azure_blob_path(**config)
    result = subprocess.run(["azcopy", "ls", azure_path], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    all_checkpoints_names = []
    for out_str in output.split("\n"):
        if "checkpoint" not in out_str:
            continue
        else:
            # strip extra info ("INFO" and "Content Length..."")
            file_name = out_str.split(";")[0][6:]
            all_checkpoints_names.append(file_name)

    unique_checkpoints = list(set([cpt.split("-")[0] for cpt in all_checkpoints_names]))
    if not silent:
        logging.info(f"All checkpoints: {unique_checkpoints}")
    
    if len(unique_checkpoints) == 0:
        logging.info(f"No checkpoints at '{input_path}' on Azure! - full path {azure_path}")
    
    checkpoint_id_to_checkpoint_base_mapper = {}
    for checkpoint in unique_checkpoints:
        cpt_id = checkpoint.split("_")[-1]
        if cpt_id.isdigit():
            checkpoint_id_to_checkpoint_base_mapper[int(cpt_id)] = checkpoint

    available_checkpoint_ids_on_azure = sorted((list(checkpoint_id_to_checkpoint_base_mapper.keys())))

    processed_checkpoint_ids = []
    downloaded_checkpoint_paths = []
    for cpt_id in checkpoint_ids:
        # download the last checkpoint
        if cpt_id < 0 and len(available_checkpoint_ids_on_azure) > 0:
            cpt_id = available_checkpoint_ids_on_azure[cpt_id]
        
        if checkpoint_id_to_checkpoint_base_mapper.get(cpt_id) is not None:
            checkpoint = checkpoint_id_to_checkpoint_base_mapper[cpt_id]
            # check if it exists in the output path
            azure_path = build_azure_blob_path(**config, read_only_sub_dir=True)
            if not os.path.exists(os.path.join(full_model_path, checkpoint)):
                # This will not be silent because it takes a lot of time and is worth having feedback
                logging.info(f"Downloading from {azure_path} to {os.path.join(full_model_path, checkpoint)}")
                checkpoint_out_path = f"{os.path.join(full_model_path, checkpoint)}"
                result = subprocess.run(["azcopy", "cp", "--include-pattern", f"{checkpoint}-*", azure_path, checkpoint_out_path])
                downloaded_checkpoint_paths.append(checkpoint_out_path)
            else:
                if not silent:
                    logging.warning(f"Path already exists: {os.path.join(full_model_path, checkpoint)}")
            processed_checkpoint_ids.append(cpt_id)

    return available_checkpoint_ids_on_azure, azure_path, downloaded_checkpoint_paths


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument(
        "-c",
        "--cluster",
        default="azure",
        help="Set this to the cluster type from which we want to download the model checkpoints",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        help="The path where we want to place the downloaded model checkpoints; give the parent directory of the directory containing all the checkpoints",
    )

    parser.add_argument(
        "--config-file",
        help="The path to the config file containing information of azure blob settings",
    )

    parser.add_argument(
        "--checkpoint-id-filter",
        default="0-10k:1k,10k-20k:5k,20k-100k:20k,100k-1000k:50k"
    )

    parser.add_argument(
        "--last-checkpoint",
        action="store_true",
        help="Downloads last checkpoint if it does not exist.",
    )

    args = parser.parse_args()

    if args.cluster == "azure":
        if args.last_checkpoint:
            checkpoint_ids = [-1]
        else:
            checkpoint_ids = get_checkpoint_ids_from_text(args.checkpoint_id_filter)
        
        download_checkpoints_from_azure_blob(args.config_file, 
                                             args.output_path, 
                                             checkpoint_ids, 
                                             silent=False, 
                                             append_model_name_to_out_path=True)
    else:
        raise NotImplementedError(f"Downloading from {args.cluster} cluster is not implemented yet!")
