import argparse
import collections
import pandas as pd

from examples.few_shot.tasks import FEW_SHOT_TASKS_REGISTRY, get_all_tasks
from examples.few_shot.scripts.sanity_check_results import get_result_id


def sanity_check_multilingual_results(input_files):
    result_languages = collections.defaultdict(list)
    for in_tsv in input_files:
        with open(in_tsv) as f:
            df = pd.read_csv(f, sep="\t")
            for idx, row in df.iterrows():
                result_id = get_result_id(row)
                language = row['language']
                result_languages[result_id].append(language)

    for result_id in result_languages:
        task_name = result_id.split('_tmp.')[0].split('_task.')[1]
        r_langs = sorted(result_languages[result_id])
        task_langs = sorted(FEW_SHOT_TASKS_REGISTRY[task_name].get_supported_languages())
        if r_langs == task_langs:
            print(f"{result_id} language check passed")
        else:
            for lang in r_langs:
                assert(lang in task_langs), "Error: unexpected language \"{lang}\" for {result_id}"
            for lang in task_langs:
                if lang not in r_langs:
                    print(f"* {lang} missing from {result_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sanity check multilingual evaluation result table")
    parser.add_argument(
        "-i", 
        "--input-files",
        default=[
            "results.tsv"
        ],
        nargs="+",
        help="List of multilingual result .tsv files to be sanity checked"
    )
    args = parser.parse_args()

    sanity_check_multilingual_results(args.input_files)