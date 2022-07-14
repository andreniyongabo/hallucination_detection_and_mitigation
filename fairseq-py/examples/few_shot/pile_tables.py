import pandas as pd
from pathlib import Path

def parse_preprocess_log(pth):
    res = []
    for ln in Path(pth).open().readlines():
        if 'sents' not in ln: continue
        stats = {}
        tok_count = ln.split('tokens,')[0].split(', ')[-1].strip()
        stats['tok_count'] = int(tok_count)
        common_prefix = '[None] data/'
        pth = ln.split(':')[0][len(common_prefix):]
        stats['dataset'] = Path(pth).parent.name
        stats['split'] = 'valid' if 'valid' in pth else 'test'

        res.append(stats)
    return res


def token_count_table(pile_dir='/private/home/sshleifer/ThePile/spm_cc100_xl/'):
    res = []
    for path in Path(pile_dir).glob('*/preprocess.log'):
        res.extend(parse_preprocess_log(path))
    tok_count_sp = pd.DataFrame(res)
    return tok_count_sp
