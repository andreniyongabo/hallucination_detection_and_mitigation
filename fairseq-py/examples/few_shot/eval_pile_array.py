from pathlib import Path

import submitit
from examples.few_shot.models import MULTI_LM_CC100_COMBINED_ROBERTA
from examples.few_shot.eval_pile import eval_pile
from fire import Fire
import itertools

ALL_PILE_DATASETS = ['Bibliotik', 'BookCorpus', 'CommonCrawl', 'DM_Mathematics', 'Enron_Emails', 'EuroParl',
                     'FreeLaw', 'Github', 'Gutenberg_PG-19', 'HackerNews', 'NIH_ExPorter', 'OpenSubtitles',
                     'OpenWebText2', 'PhilPapers', 'PubMed_Abstracts', 'PubMed_Central', 'StackExchange',
                     'USPTO', 'Ubuntu_IRC', 'Wikipedia_en', 'YoutubeSubtitles']

def get_args_to_run():
    mnames = list(MULTI_LM_CC100_COMBINED_ROBERTA.keys())
    small_datasets = ['EuroParl', 'Enron_Emails', 'YoutubeSubtitles',  'Wikipedia_en']
    # datasets
    product = list(itertools.product(mnames, small_datasets))
    models = [x[0] for x in product]
    datasets = [x[1] for x in product]
    return models, datasets


def run_job():
    logs = f'submitit_logs/'
    Path(logs).mkdir(exist_ok=True, parents=True)
    executor = submitit.AutoExecutor(folder=str(logs))
    executor.update_parameters(
        tasks_per_node=8,
        nodes=2,
        slurm_partition='learnaccel',
        mem_gb=400,
        
        gpus_per_node=8,
        slurm_constraint='volta32gb',
        slurm_time=4320,
        slurm_array_parallelism=12,
    )
    models, datasets = get_args_to_run()
    job = executor.map_array(
        eval_pile,
        models, datasets
    )
    print(f'submitted {len(models)} jobs.')


if __name__ == '__main__':
    Fire(run_job)
