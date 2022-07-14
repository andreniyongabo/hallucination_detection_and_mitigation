import argparse
import collections
import pandas as pd


def get_result_id(res, 
                  keys=[
                      'task',
                      'model_name', 
                      'template', 
                      'train_set',
                      'eval_set',
                      'calibration',
                      'nb_few_shot_samples',
                  ],
                  result_metrics='accuracy::mean',
                  ignore_metrics=None):
    metrics = []
    for key in keys:
        if key == 'model_name' and 'step' in ignore_metrics:
            metrics.append((key, res[key].split('__step')[0]))
        else:
            metrics.append((key, res[key]))
    result_id = '_'.join([f'{k}.{v}' for k, v in metrics]) + f'_{result_metrics}'
    return result_id


def sanity_check_results(in_tsv):
    eval_grid = {
        "6.7B_gpt3_setting": {
            "steps": [10000, 30000, 50000, 70000, 90000, 110000, 130000, 143050],
        },
        "dense_7.5B_lang30_new_cc100_xl_unigram": {
            "steps": [5000, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 238000]
        }
    }

    result_group = collections.defaultdict(list)
    with open(in_tsv) as f:
        df = pd.read_csv(f, sep="\t")
        for _, row in df.iterrows():
            result_id = get_result_id(row, ignore_metrics=["step"])
            model_name = row['model_name']
            step = int(model_name.split('__step')[1])
            result_group[result_id].append(step)

    for model_name in eval_grid:
        err_detected = False

        # steps to evaluate
        steps_to_eval = eval_grid[model_name]["steps"]

        # steps evaluated
        for result_id in result_group:
            if f'model_name.{model_name}' in result_id:
                steps_evaluated = sorted(result_group[result_id])
                for step in steps_to_eval:
                    if step not in steps_evaluated:
                        print(f'* missing step {step} for {result_id}')
                        err_detected = True
                for step in steps_evaluated:
                    if step not in steps_to_eval:
                        print(f'* extra step {step} for {result_id}')
                        err_detected = True

            if not err_detected:
                print(f"TEST PASSED: {result_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sanity check en evaluation result table w.r.t. different critera")
    parser.add_argument(
        "-i", 
        "--input-file",
        default="results.tsv",
        help="A result .tsv files to be sanity checked"
    )
    args = parser.parse_args()

    sanity_check_results(args.input_file)
    