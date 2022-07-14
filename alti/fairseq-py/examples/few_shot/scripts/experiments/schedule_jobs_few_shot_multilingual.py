# fmt: off
import argparse 
import sys
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *
from examples.few_shot.tasks import PawsXTask, StoryClozeTask, XCOPATask, XNLITask

if __name__ == "__main__":
    """
    The current script aims at executing experiments that will be used\
        to explore the prompt PPL to target performance. 
    
    Selected tasks: 
        - Multi-choice multilingual
            - XCOPA, EXAMS, XNLI, PAWSX

        - Generation
            - Translation tasks

    Setting:
        - Scoring - `mean`
        - nshot - 0, 1, 4, 32, 128. The initial experiments with 0-shot has shown strong correlation and want to see if it works for > 0 shot. 
        - Trials - 5 trials are used to reduce variance.

    Commands:
        - Run 
            
            # Base evaluation before the big run. We decided to focus on a few tasks. 
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v0 -m dense_vs_moe_lang16 --nshot 0 --local --dry-run

            # cross lingual settings
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t paws_x_cross_lingual xnli_cross_lingual xcopa_cross_lingual -m 125M_gpt3_setting -o /checkpoint/${USER}/few_shot/multilingual_crosslingual --nshot 32 --local --dry-run
            
            #xnli 
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_mt -m dense_lang16 --nshot 0 -o /checkpoint/${USER}/few_shot/xnli_experimental --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli -m dense_lang16 --nshot 0 -o /checkpoint/${USER}/few_shot/xnli_experimental --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli -m dense_lang16 --nshot 0 -o /checkpoint/${USER}/few_shot/xnli_experimental --local --dry-run

            # xnli with avialable multilingual models 
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli -m 2021_09_multilingual_eval_v1 --nshot 0 -o /checkpoint/${USER}/few_shot/2021_09_multilingual_eval_v1_full --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t pawsx_mt_all -m 2021_09_multilingual_eval_v1 --nshot 0 -o /checkpoint/${USER}/few_shot/2021_09_multilingual_eval_v1_full --local --dry-run
            
            # with calib
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t pawsx_mt_all_calib -m 2021_09_multilingual_eval_v1 --nshot 0 -o /checkpoint/${USER}/few_shot/2021_09_multilingual_eval_v1_full_calib --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli_calib -m 2021_09_multilingual_eval_v1 --nshot 0 -o /checkpoint/${USER}/few_shot/2021_09_multilingual_eval_v1_full_calib --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xwinograd -m dense_lang16 --nshot 0 -o /checkpoint/${USER}/few_shot/2021-09-15-xwinograd --local --dry-run
            
            # debug calib
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t pawsx_mt_all_calib -m 125M_gpt3_setting --nshot 0 -o /checkpoint/${USER}/few_shot/2021_09_multilingual_eval_v1_debug --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli_calib -m 125M_gpt3_setting --nshot 0 -o /checkpoint/${USER}/few_shot/2021_09_multilingual_eval_v1_debug --local --dry-run

            #xnli en-only - experiments with dense models
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli_en -m 1.3B_gpt3_setting_checkpoints --nshot 0 -o /checkpoint/${USER}/few_shot/multilingual_xnli_1.3B_gpt3_setting_checkpoints --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli_en -m gpt3_setting --nshot 0 -o /checkpoint/${USER}/few_shot/multilingual_xnli_1.3B_gpt3_setting_checkpoints --local --dry-run
            
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t pawsx_mt_en -m 1.3B_gpt3_setting_checkpoints --nshot 0 -o /checkpoint/${USER}/few_shot/1.3B_gpt3_setting_checkpoints
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t pawsx_mt_en -m gpt3_setting --nshot 0 -o /checkpoint/${USER}/few_shot/multilingual_xnli_gpt3_setting --local --dry-run
            
            # mLama
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t mlama_all_top5 -m dense_lang16 --nshot 0  -o /checkpoint/${USER}/few_shot/2021-09-17-mlama --local --dry-run        
    
            # full evaluation
            ## multilingual_bigrun_eval_v1
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v1 -m dense_7.5B_lang30_new_cc100_xl_unigram__step00150000 --nshot 0 -o /checkpoint/${USER}/few_shot/dense_7.5B_lang30_new_cc100_xl_unigram_mutli_tasks_v1 --slurm-partition devaccel,learnaccel --dry-run --local 
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v1 -m model_moe_200B_lang30_setting_checkpoints --nshot 0 -o /checkpoint/${USER}/few_shot/dense_7.5B_lang30_new_cc100_xl_unigram_mutli_tasks_v1 --slurm-partition xlmg,devaccel --dry-run --local

            ## multilingual_bigrun_eval_v1_extended

            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v1_extended -m dense_lang16 l28_dense_1b24_2cc100_combined_roberta --nshot 0 -o /checkpoint/${USER}/few_shot/2021-09-15-multilingual_bigrun_eval_v1_extended --n-eval-samples 100 --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v1_extended -m dense_lang16 l28_dense_1b24_2cc100_combined_roberta --nshot 1 -o /checkpoint/${USER}/few_shot/2021-09-15-multilingual_bigrun_eval_v1_extended
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v1_extended -m dense_lang16 l28_dense_1b24_2cc100_combined_roberta dense_lang134_cc100_xl_supershard_unigram_alpha_0.7 dense_lang134_new_cc100_xl_old_bpe dense_lang134_new_cc100_xl_unigram --nshot 1 -o /checkpoint/${USER}/few_shot/2021-09-15-multilingual_bigrun_eval_v1_extended --n-eval-samples 100 --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t pawsx_mt_all -m dense_lang16 --nshot 0 -o /checkpoint/${USER}/few_shot/2021-09-15-multilingual_bigrun_eval_v1_extended --n-eval-samples 100 --local --dry-run

            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t multilingual_bigrun_eval_v1_extended -m dense_7.5B_lang30_new_cc100_xl_unigram_cpts_selected --nshot 0 -o /large_experiments/xlmg/results/multilingual/dense_7_5B_30lang_cpts --dry-run --local

            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t xnli_generativenli_calib -m dense_lang16 --nshot 0 -o /checkpoint/${USER}/few_shot/debug  --n-eval-samples 10 --local  --dry-run 

            # storycloze crosslingual
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t storyclose_cross_lingual_val2016_split_20_80 -m dense_7.5B_lang30_new_cc100_xl_unigram__step00238000 --nshot 32 -o /large_experiments/xlmg/results/multilingual/dense_7.5B_crosslingual --slurm-array-parallelism 30 --local  --dry-run 


        
        - Run on salloc with moe
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py -t dense_vs_moe_lang16 -m dense_vs_moe_lang16 --nshot 0 --local --dry-run

    """

    # task settings
    available_tasks_settings = default_tasks_settings.copy()  # See examples/few_shot/scripts/experiments/schedule_jobs_few_shot.py
    available_tasks_settings.update({
        # put setting overrides here
        # task run key: (task run name - used for results directory prefix, tasks list List[str], evaluation params: Dict[param_name, param_value] -- these are the same gpt3_eval input params)
        "pawsx_mt": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_mt"]}),
        "pawsx_mt_en": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_mt"], "pawsx_languages": ["en"]}),
        "pawsx_mt_calib": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_mt"], "calibrator_name": "average_option", "pawsx_calibration_options": ["sentence1::"], }),
        "pawsx_generativenli__en": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_generativenli__en"], "pawsx_train_set": "dev", "pawsx_eval_set": ["test"]}),
        "pawsx_generativenli__en_dev": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_generativenli__en"], "pawsx_train_set": "dev", "pawsx_eval_set": ["dev"]}),
        "pawsx_generativenli__mt_dev": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_generativenli_mt"], "pawsx_train_set": "dev", "pawsx_eval_set": ["dev"]}),

        # xcopa
        "xcopa": ("xcopa", ["xcopa"], {"xcopa_train_set": "val", "xcopa_eval_set": ["test"]}),
        "xcopa__en": ("xcopa__en", ["xcopa"], {"xcopa_template": "xcopa__en", "xcopa_train_set": "val", "xcopa_eval_set": ["test"]}),
        "xcopa__en_dev": ("xcopa__en", ["xcopa"], {"xcopa_template": "xcopa__en", "xcopa_train_set": "val", "xcopa_eval_set": ["val"]}),
        "xcopa_mt_dev": ("xcopa_mt", ["xcopa"], {"xcopa_template": "xcopa_mt", "xcopa_train_set": "val", "xcopa_eval_set": ["val"]}),
        ## The simple setting is shown to be weaker than using verbal templates
        # "xcopa__simple": ("xcopa__simple", ["xcopa"], {"xcopa_template": ["xcopa_simple"], "xcopa_train_set": "val", "xcopa_eval_set": ["val"]}),

        # xnli
        "xnli_mt": ("xnli_mt", ["xnli"], {"xnli_template": ["xnli_mt"]}),
        "xnli_generativenli__en": ("xnli", ["xnli"], {"xnli_template": ["xnli_generativenli__en"], "xnli_train_set": "dev", "xnli_eval_set": ["test"]}),
        "xnli_generativenli__en_dev": ("xnli", ["xnli"], {"xnli_template": ["xnli_generativenli__en"], "xnli_train_set": "dev", "xnli_eval_set": ["dev"]}),
        "xnli_generativenli_mt_dev": ("xnli", ["xnli"], {"xnli_template": ["xnli_generativenli_mt"], "xnli_train_set": "dev", "xnli_eval_set": ["dev"]}),
        "xnli_generativenli_ht_dev": ("xnli", ["xnli"], {"xnli_template": ["xnli_generativenli_ht"], "xnli_train_set": "dev", "xnli_eval_set": ["dev"]}),

        # mLama tasks
        "mlama_googlere": ("mlama_googlere", ["mlama_googlere"], {}),
        "mlama_trex": ("mlama_trex", ["mlama_trex"], {}),
        ## Fast version with only 5 cands used as reference. Using the full tasks would be super slow!
        "mlama_all_top5": ("mlama_all_top5", ["mlama_googlere", "mlama_trex"], {"max_cands": 5}),
        "mlama_all_top10": ("mlama_all_top10", ["mlama_googlere", "mlama_trex"], {"max_cands": 10}),
    })

    # experimental
    available_tasks_settings.update({
        # XNLI experimental
        "xnli_generativenli": ("xnli_experimental", ["xnli"], {"xnli_template": ["xnli_generativenli_sentence_mt", "xnli_generativenli_mt", "xnli_generativenli_ht", "xnli_generativenli_sentence__en", "xnli_generativenli__en"]}),
        "xnli_generativenli_en": ("xnli_experimental", ["xnli"], {"xnli_languages": ["en"], "xnli_template": ["xnli_generativenli_sentence_mt", "xnli_generativenli_mt", "xnli_generativenli_sentence__en", "xnli_generativenli__en",  "generativenli"]}),
        "xnli_generativenli_200": ("xnli_experimental_200", ["xnli"], {"n_eval_samples": 10, "xnli_template": ["xnli_generativenli_mt", "xnli_generativenli_sentence__en", "xnli_generativenli__en",  "generativenli"]}),
        "xnli_generativenli_mt": ("xnli_experimental", ["xnli"], {"xnli_template": ["xnli_generativenli_mt"]}),

        "xnli_generativenli_ht_calib_en": ("xnli_experimental", ["xnli"], {"xnli_template": ["xnli_generativenli_ht"], 
                                                            "xnli_languages": ["en"],
                                                            "calibrator_name": "average_option",
                                                            "xnli_calibration_options": ["sentence1::"],}),

        "xnli_generativenli_calib": ("xnli_experimental", ["xnli"], {"xnli_template": ["xnli_generativenli_sentence_mt", "xnli_generativenli_mt", "xnli_generativenli_ht", "xnli_generativenli_sentence__en", "xnli_generativenli__en"], 
                                                            "calibrator_name": "average_option",
                                                            "xnli_calibration_options": ["sentence1::"],}),

        # pawsx experimental
        "pawsx_mt_all": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_mt", "pawsx_generativenli_mt", "pawsx_generativenli_ht", "pawsx_conditional_mt"]}),
        "pawsx_mt_all_calib": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_mt", "pawsx_generativenli_mt", "pawsx_generativenli_ht", "pawsx_conditional_mt"],
                                                    "calibrator_name": "average_option",
                                                    "pawsx_calibration_options": ["sentence1::"], }),
        
        "xwinograd": ("xwinograd", ["xwinograd"], {"xwinograd_template": ["winograd"]}),
        "storycloze_multilingual_dev": ("storycloze", ["storycloze"], {"storycloze_template": ["storycloze"], "storycloze_train_set": "val2016", "storycloze_eval_set": ["val2016"]}),
        "storycloze_multilingual_dev_eu": ("storycloze", ["storycloze"], {"storycloze_template": ["storycloze"], "storycloze_train_set": "val2016", "storycloze_eval_set": ["val2016"], "storycloze_languages": ["eu"]}),
        "storycloze_multilingual_dev_id": ("storycloze", ["storycloze"], {"storycloze_template": ["storycloze"], "storycloze_train_set": "val2016", "storycloze_eval_set": ["val2016"], "storycloze_languages": ["id"]}),
        "storycloze_multilingual_dev_my": ("storycloze", ["storycloze"], {"storycloze_template": ["storycloze"], "storycloze_train_set": "val2016", "storycloze_eval_set": ["val2016"], "storycloze_languages": ["my"]}),
        "storycloze_multilingual_val2016_split_20_80": ("storycloze", ["storycloze"], {"storycloze_template": ["storycloze"], "storycloze_train_set": "val2016_split_20_80_train", "storycloze_eval_set": ["val2016_split_20_80_eval"]}),
    })

    task_run_groups = default_task_run_groups.copy()
    task_run_groups = {
        "multilingual_multichoice": ["xcopa", "xnli", "exams", "pawsx", "mlama_all_top5"],
        "multilingual_bigrun_eval_v0": ["xcopa", "xnli", "pawsx", "mlama_all_top5"],
        "multilingual_bigrun_eval_v1": ["storycloze_multilingual_val2016_split_20_80", "xcopa__en", "xnli_generativenli__en", "pawsx_generativenli__en", "xwinograd"], #, "mlama_all_top10", "mlama_all_top5"],
        "multilingual_bigrun_eval_v1_extended": ["storycloze_multilingual_dev", "pawsx_mt_all_calib", "xnli_generativenli_calib", "xcopa", "xwinograd"],
        "translation": ['wmt14fren', 'wmt14enfr', 'wmt16deen', 'wmt16ende', 'wmt16roen', 'wmt16enro'],
        "pawsx_default_vs_gt_calib": ["pawsx_default_calib", "pawsx_mt_calib"]
    }
    
    # PAWS X add crosslingual few-shot setting
    new_task_group_name = "paws_x_cross_lingual"
    new_task_group_task_settings = []
    for eval_lang_code in PawsXTask.get_supported_languages():  # dev is the default train
        for train_lang_code in PawsXTask.get_supported_languages():  # dev is the default train
            setting_key = f"paws_x_cross_lingual_train_{train_lang_code}_eval_{eval_lang_code}"
            available_tasks_settings[setting_key] = ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_mt"], "pawsx_train_lang": train_lang_code, "pawsx_languages": [eval_lang_code]})
            new_task_group_task_settings.append(setting_key)
    task_run_groups[new_task_group_name] = new_task_group_task_settings

    # XNLI add crosslingual few-shot setting
    new_task_group_name = "xnli_cross_lingual"
    new_task_group_task_settings = []
    for eval_lang_code in XNLITask.get_supported_languages():  # dev is the default train
        for train_lang_code in XNLITask.get_supported_languages():  # dev is the default train
            setting_key = f"xnli_cross_lingual_train_{train_lang_code}_eval_{eval_lang_code}"
            available_tasks_settings[setting_key] = ("xnli", ["xnli"], {"xnli_template": ["xnli_mt"], "xnli_train_lang": train_lang_code, "xnli_languages": [eval_lang_code]})
            new_task_group_task_settings.append(setting_key)
    task_run_groups[new_task_group_name] = new_task_group_task_settings

    # XCOPA add crosslingual few-shot setting
    new_task_group_name = "xcopa_cross_lingual"
    new_task_group_task_settings = []
    for eval_lang_code in XCOPATask.get_supported_languages():  # dev is the default train
        for train_lang_code in XCOPATask.get_supported_languages():  # dev is the default train
            setting_key = f"xcopa_cross_lingual_train_{train_lang_code}_eval_{eval_lang_code}"
            available_tasks_settings[setting_key] = ("xcopa", ["xcopa"], {"xcopa_template": ["xcopa"], "xcopa_train_lang": train_lang_code, "xcopa_languages": [eval_lang_code]})
            new_task_group_task_settings.append(setting_key)
    task_run_groups[new_task_group_name] = new_task_group_task_settings        

    # StoryClose crosslingual few-shot setting - val2016_split_20_80
    new_task_group_name = "storyclose_cross_lingual_val2016_split_20_80"
    new_task_group_task_settings = []
    for eval_lang_code in StoryClozeTask.get_supported_languages():  # dev is the default train
        for train_lang_code in StoryClozeTask.get_supported_languages():  # dev is the default train
            setting_key = f"storyclose_cross_lingual_val2016_split_20_80_train_{train_lang_code}_eval_{eval_lang_code}"
            available_tasks_settings[setting_key] = TaskSchedulingSetting(
                                                        job_name_prefix="storyclose_cross_lingual_val2016_split_20_80",
                                                        tasks=["storycloze"],
                                                        eval_params={
                                                            "storycloze_train_set": "val2016_split_20_80_train", 
                                                            "storycloze_eval_set": ["val2016_split_20_80_eval"],
                                                            "storycloze_train_lang": train_lang_code,
                                                            "storycloze_languages": [eval_lang_code],
                                                        }
            )
            new_task_group_task_settings.append(setting_key)
    task_run_groups[new_task_group_name] = new_task_group_task_settings


    # model settings
    available_model_settings, model_run_groups = get_extended_default_model_settings_and_groups()
    available_model_settings.update({
        # model_name, job_params, custom params, nodes, gpus_per_node, ntasks_per_node, cpus_per_task, max parallel jobs in this pool
        "dense_lang16": ("dense_lang16", {}, {"train_sep": "\n"}, 1, 1, 1, 8, 3),
        "moe_128exp_lang16": ("moe_128exp_lang16", {}, {"train_sep": "\n"}, 2, 8, 1, 8, 1),
        # experimental 
        "l28_dense_1b24_2cc100_combined_roberta": ("l28_dense_1b24_2cc100_combined_roberta", {}, {"train_sep": "\n", "replace_newline_with_eos": True}, 1, 1, 1, 8, 3),
        "l28_64e_1b24_top1_cc100_combined_roberta": ("l28_64e_1b24_top1_cc100_combined_roberta", {}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875}, 2, 8, 1, 8, 1),
        "l28hi_64e_top2_l24": ("l28hi_64e_top2_l24", {}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875}, 8, 8, 1, 8, 1),
        
        "moe_64exp_lang1_cc100_combined_roberta": ("moe_64exp_lang1_cc100_combined_roberta_0921", {}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875}, 2, 8, 1, 8, 1),
        "moe_64exp_lang32_cc100_combined_roberta_1.0": ("moe_64exp_lang32_cc100_combined_roberta_1.0_0921", {}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875}, 2, 8, 1, 8, 1),
        "moe_64exp_lang32_cc100_combined_roberta_1.0_adapt": ("moe_64exp_lang32_cc100_combined_roberta_1.0_adapt_0921", {}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875}, 1, 8, 1, 8, 1),
    
        "dense_lang134_cc100_xl_supershard_unigram_alpha_0.7": ("dense_lang134_cc100_xl_supershard_unigram_alpha_0.7", {}, {"train_sep": "\n", "replace_newline_with_eos": True}, 1, 1, 1, 8, 3),
        "dense_lang134_new_cc100_xl_old_bpe": ("dense_lang134_new_cc100_xl_old_bpe", {}, {"train_sep": "\n", "replace_newline_with_eos": True}, 1, 1, 1, 8, 3),
        "dense_lang134_new_cc100_xl_unigram": ("dense_lang134_new_cc100_xl_unigram", {}, {"train_sep": "\n", "replace_newline_with_eos": True}, 1, 1, 1, 8, 3),
    })
    

    model_run_groups.update({
        "dense_vs_moe_lang16": ["dense_lang16", "moe_128exp_lang16"],

        "2021_09_multilingual_eval_v1": [
            "l28_dense_1b24_2cc100_combined_roberta", 
            "l28_64e_1b24_top1_cc100_combined_roberta",
            "moe_64exp_lang1_cc100_combined_roberta",
            "moe_64exp_lang32_cc100_combined_roberta_1.0",
            "moe_64exp_lang32_cc100_combined_roberta_1.0_adapt",
            "l28hi_64e_top2_l24",
        ],

        "dense_7.5B_lang30_new_cc100_xl_unigram_cpts_selected": [
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00005000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00030000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00060000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00090000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00120000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00150000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00180000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00210000",
            "dense_7.5B_lang30_new_cc100_xl_unigram__step00238000",
        ],

        "moe_200B_lang30_new_cc100_xl_unigram_cpts_selected": [
            "moe_200B_lang30_new_cc100_xl_unigram__step00048000",
            "moe_200B_lang30_new_cc100_xl_unigram__step00060000",
            "moe_200B_lang30_new_cc100_xl_unigram__step00096000"
        ]
    })
    

    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for multilingual experiments.")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/multilingual")
    arg_modify_default(parser, "scoring", "mean")
    arg_modify_default(parser, "slurm_partition", "learnaccel")
    arg_modify_default(parser, "nb_few_shot_samples_values", [0, 1])
    arg_modify_default(parser, "num_trials", 5)

    # set default dir for results
    args = parser.parse_args()

    #args.dry_run = True
    #args.local = True

    print("Arguments:")
    print(args)
       
    # schedule jobs -- see the function documenation to learn the order of param updates.
    schedule_experiment_jobs(args, 
                             task_run_groups, available_tasks_settings, 
                             model_run_groups, available_model_settings,
                             custom_base_run_args = {}, 
                             custom_override_run_args = {}
                            )

    print_display_results_command(args)
    sys.exit(0) # sometimes srun hangs and calling sys.exit(0) exits properly
