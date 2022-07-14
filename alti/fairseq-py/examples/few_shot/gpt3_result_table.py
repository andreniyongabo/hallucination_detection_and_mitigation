import fire
import pandas as pd
from pathlib import Path
from fairseq.file_io import load_json

SUPERGLUE_TASKS_PLUS = ['copa', 'wic', 'boolq', 'cb', 'rte', 'wsc', 'record', 'multirc', 'openbookqa', 'arceasy', 'arcchallenge', 'openbookqa']

def make_halil_table(result_dir='/large_experiments/xlmg/models/sshleifer/few_shot_results/', keep_tasks=None):
    result_paths = Path(result_dir).glob('*/*results.json')
    records = []
    for p in result_paths:
        data = load_json(p)
        acc = data['accuracy']
        keys_to_keep = ['model_name', 'task', 'nb_few_shot_samples',]
        record = {k: data[k] for k in keys_to_keep}
        record.update(acc)
        records.append(record)
    tab = pd.DataFrame(records)
    if keep_tasks is not None:
        tab = tab[tab.task.isin(keep_tasks)]
    else:
        pass
        # tab = tab[~(tab.task.isin(['xcopa', 'exams']))]
    tab['setting'] = tab['task'] + '-' + tab['nb_few_shot_samples'].astype(str)
    tab = tab.drop(['scores', 'mean_confidence_interval', 'nb_few_shot_samples', 'task'], 1, errors='ignore')

    tab = tab.rename(columns={
        #'nb_few_shot_samples': 'nshot',
        'setting': 'task',
        'model_name': 'm',
        #'mean': 'acc_mean',
        #'std': 'acc_std',
    })
    t = tab.set_index(['m', 'task'])
    t.columns.name = 'metric'
    t = t.stack().unstack(['task', 'metric']).round(2).sort_index(axis=1)
    return t


if __name__ == '__main__':
    fire.Fire(make_halil_table)
