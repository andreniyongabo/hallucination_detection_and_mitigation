import functools
import pandas as pd


xlmg_language_resource_brackets = {
    "high": {  # 10 languages
        "en_XX",
        "es_XX",
        "de_DE",
        "fr_XX",
        "it_IT",
        "zh_CN",
        "ru_RU",
        "pt_XX",
        "ar_AR",
        "nl_XX",
    },
    "medium": {  # 26 languages
        "id_ID",
        "fa_IR",
        "el_GR",
        "vi_VN",
        "tr_TR",
        "pl_PL",
        "uk_UA",
        "ja_XX",
        "jp_XX",
        "zh_TW",
        "sv_SE",
        "cs_CZ",
        "ro_RO",
        "th_TH",
        "hu_HU",
        "ko_KR",
        "no_XX",
        "bg_BG",
        "fi_FI",
        "da_DK",
        "hr_HR",
        "he_IL",
        "ms_MY",
        "sk_SK",
        "hi_IN",
        "ca_ES",
        "lt_LT",
    },
    "low": {  # 32 languages
        "sl_SI",
        "et_EE",
        "ta_IN",
        "sr_RS",
        "bn_IN",
        "lv_LV",
        "ka_GE",
        "tl_XX",
        "hy_AM",
        "sq_AL",
        "ml_IN",
        "ur_PK",
        "kk_KZ",
        "hi_IN_rom",
        "mk_MK",
        "be_BY",
        "te_IN",
        "ne_NP",
        "mn_MN",
        "is_IS",
        "si_LK",
        "kn_IN",
        "mr_IN",
        "bs_BA",
        "sw_KE",
        "af_ZA",
        "gl_ES",
        "gu_IN",
        "km_KH",
        "la_VA",
        "ps_AF",
        "eu_ES",
        "cb_IQ",
    },
    "extremely-low": {  # 65 languages
        "so_SO",
        "ar_AR_rom",
        "uz_UZ",
        "my_MM_zaw",
        "cy_GB",
        "jv_ID",
        "ky_KG",
        "pa_IN",
        "am_ET",
        "eo_EO",
        "or_IN",
        "ur_PK_rom",
        "lo_LA",
        "my_MM",
        "bn_IN_rom",
        "ha_NG",
        "ga_IE",
        "mg_MG",
        "sa_IN",
        "ku_TR",
        "sd_PK",
        "ht_HT",
        "te_IN_rom",
        "su_ID",
        "cx_PH",
        "ta_IN_rom",
        "az_IR",
        "ti_ET",
        "br_FR",
        "yo_NG",
        "fy_NL",
        "zu_ZA",
        "as_IN",
        "gd_GB",
        "xh_ZA",
        "om_KE",
        "qh_PH",
        "ci_IT",
        "lg_UG",
        "ig_NG",
        "ns_ZA",
        "tn_BW",
        "wo_SN",
        "ff_NG",
        "gn_PY",
        "qf_CM",
        "ln_CD",
        "qm_AO",
        "qd_MM",
        "qj_ML",
        "bm_ML",
        "qx_KE",
        "qy_KE",
        "q3_CV",
        "kg_AO",
        "ql_ML_rom",
        "qa_MM",
        "qi_PH_rom",
        "ss_SZ",
        "qw_KE",
        "qu_PE",
        "qp_AO",
        "qq_KE",
    },
}

_all_eval_tasks = [
    "arcchallenge",
    "arceasy",
    "copa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "storycloze",
    "xnli",
    "xcopa",
    "xwinograd",
    "pawsx",
]

_all_en_eval_tasks = [
    "arcchallenge",
    "arceasy",
    "copa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "storycloze",
]

_all_multi_eval_tasks = ["storycloze", "xnli", "xcopa", "xwinograd", "pawsx"]

multi_eval_tasks_langs = {
    "storycloze": {"en", "ru", "zh", "ar", "sw", "hi", "es", "my", "id", "eu", "te"},
    "xnli": {
        "en",
        "fr",
        "es",
        "de",
        "el",
        "bg",
        "ru",
        "tr",
        "ar",
        "vi",
        "th",
        "zh",
        "hi",
        "sw",
        "ur",
    },
    "xcopa": {"et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"},
    "xwinograd": {"en", "jp", "pt", "ru"},  # {'en', 'jp', 'pt', 'ru', 'fr', 'zh'},
    "pawsx": {"de", "en", "es", "fr", "ja", "ko", "zh"},
}

# filtering conditions
def all_eval_tasks(df):
    return functools.reduce(
        lambda x, y: x | y, [df["task"] == t for t in _all_eval_tasks]
    )


def all_en_eval_tasks(df):
    return functools.reduce(
        lambda x, y: x | y, [df["task"] == t for t in _all_en_eval_tasks]
    )


def all_multi_eval_tasks(df):
    return functools.reduce(
        lambda x, y: x | y, [df["task"] == t for t in _all_multi_eval_tasks]
    )


def valid_settings(df):
    return df["template"] != "xcopa_simple"


def eval_settings(option, df):
    if option == "best":
        return (
            (df["task"] != "openbookqa")
            | (df["run_params::scoring"] == "unconditional-norm")
        ) & ((df["task"] != "winogrande") | (df["run_params::scoring"] == "suffix"))
    elif option == "default":
        return (
            (df["task"] != "openbookqa")
            | (df["run_params::scoring"] != "unconditional-norm")
        ) & ((df["task"] != "winogrande") | (df["run_params::scoring"] != "suffix"))


def eval_template_selection(df):
    return (
        (df["task"] != "xcopa")
        | (df["template"] == "xcopa__en") & (df["task"] != "xnli")
        | (df["template"] == "xnli_generativenli__en")
    )


def all_checkpoints(df):
    return (
        (df.model == "dense_7.5B_lang30_new_cc100_xl_unigram") & ((df.step == 30000))
        | (df.step == 60000)
        | (df.step == 120000)
        | (df.step == 238000)
    ) | (df.model == "6.7B_gpt3_setting") & (
        (df.step == 10000)
        | (df.step == 30000)
        | (df.step == 70000)
        | (df.step == 143050)
    )


def multilingual_checkpoints(df):
    return (
        (df.model == "dense_7.5B_lang30_new_cc100_xl_unigram")
        | (df.model == "dense_1.7B_lang30_new_cc100_xl_unigram")
        | (df.model == "dense_564M_lang30_new_cc100_xl_unigram")
    )


def last_checkpoint(df):
    return (
        ((df.model == "6.7B_gpt3_setting") & (df.step == 143050))
        | ((df.model == "dense_7.5B_lang30_new_cc100_xl_unigram") & (df.step == 238000))
        | ((df.model == "dense_1.7B_lang30_new_cc100_xl_unigram") & (df.step == 58000))
        | ((df.model == "dense_564M_lang30_new_cc100_xl_unigram") & (df.step == 119209))
    )
    # | ((df.model == 'moe_200B_lang30_new_cc100_xl_unigram') & (df.step == 118000)) \


def num_few_shot_samples(df):
    return (
        (df["nb_few_shot_samples"] == 0)
        | (df["nb_few_shot_samples"] == 4)
        | (df["nb_few_shot_samples"] == 32)
        | ((df["nb_few_shot_samples"] == 128) & (df["task"] != "storycloze"))
    )


def num_few_shot_samples_all(df):
    return (
        (df["nb_few_shot_samples"] == 0)
        | (df["nb_few_shot_samples"] == 1)
        | (df["nb_few_shot_samples"] == 4)
        | (df["nb_few_shot_samples"] == 32)
        | ((df["nb_few_shot_samples"] == 128) & (df["task"] != "storycloze"))
    )


def final_eval_splits(df):
    return (
        (
            (df["task"] == "arcchallenge")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "arceasy")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "copa")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "hellaswag")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "openbookqa")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "piqa")
            & (df["eval_set"] == "valid")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "winogrande")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train_xl")
        )
        | (
            (df["task"] == "storycloze")
            & (df["language"] == "en")
            & (df["eval_set"] == "test2016")
            & (df["train_set"] == "val2016")
        )
        | (
            (df["task"] == "storycloze")
            & (df["eval_set"] == "val2016_split_20_80_eval")
            & (df["train_set"] == "val2016_split_20_80_train")
        )
        | (
            (df["task"] == "pawsx")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xcopa")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "val")
        )
        | (
            (df["task"] == "xnli")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xwinograd")
            & ((df["language"] != "fr") & (df["language"] != "zh"))
            & (df["eval_set"] == "test")
            & (df["train_set"] == "test")
        )
    )


def en_final_eval_splits(df):
    return (
        (
            (df["task"] == "arcchallenge")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "arceasy")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "copa")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "hellaswag")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "openbookqa")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "piqa")
            & (df["eval_set"] == "valid")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "winogrande")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train_xl")
        )
        | (
            (df["task"] == "storycloze")
            & (df["eval_set"] == "test2016")
            & (df["train_set"] == "val2016")
        )
        | (
            (df["task"] == "pawsx")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xcopa")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "val")
        )
        | (
            (df["task"] == "xnli")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xwinograd")
            & ((df["language"] != "fr") & (df["language"] != "zh"))
            & (df["eval_set"] == "test")
            & (df["train_set"] == "test")
        )
    )


def multi_final_eval_splits(df):
    return (
        (
            (df["task"] == "arcchallenge")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "arceasy")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "copa")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "hellaswag")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "openbookqa")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "piqa")
            & (df["eval_set"] == "valid")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "winogrande")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train_xl")
        )
        | (
            (df["task"] == "storycloze")
            & (df["eval_set"] == "val2016_split_20_80_eval")
            & (df["train_set"] == "val2016_split_20_80_train")
        )
        | (
            (df["task"] == "pawsx")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xcopa")
            & (df["language"] != "ru")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "val")
        )
        | (
            (df["task"] == "xnli")
            & (df["eval_set"] == "test")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xwinograd")
            & ((df["language"] != "fr") & (df["language"] != "zh"))
            & (df["eval_set"] == "test")
            & (df["train_set"] == "test")
        )
    )


def multi_dev_eval_splits(df):
    return (
        (
            (df["task"] == "arcchallenge")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "arceasy")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "copa")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "hellaswag")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "openbookqa")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "piqa")
            & (df["eval_set"] == "valid")
            & (df["train_set"] == "train")
        )
        | (
            (df["task"] == "winogrande")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "train_xl")
        )
        | (
            (df["task"] == "storycloze")
            & (df["eval_set"] == "val2016_split_20_80_eval")
            & (df["train_set"] == "val2016_split_20_80_train")
        )
        | (
            (df["task"] == "pawsx")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xcopa")
            & (df["eval_set"] == "val")
            & (df["train_set"] == "val")
        )
        | (
            (df["task"] == "xnli")
            & (df["eval_set"] == "dev")
            & (df["train_set"] == "dev")
        )
        | (
            (df["task"] == "xwinograd")
            & ((df["language"] != "fr") & (df["language"] != "zh"))
            & (df["eval_set"] == "test")
            & (df["train_set"] == "test")
        )
    )


def en_lang_only(df):
    return df["language"] == "en"


def get_resource_level(lang):
    for resource_level in xlmg_language_resource_brackets:
        for lang_id in xlmg_language_resource_brackets[resource_level]:
            if lang_id.startswith(f"{lang}_"):
                return resource_level


def get_task_group(task):
    task_groups = {
        "arcchallenge": "QA",
        "arceasy": "QA",
        "copa": "language modeling",
        "hellaswag": "language modeling",
        "openbookqa": "QA",
        "piqa": "QA",
        "winogrande": "Coreference",
        "storycloze": "language modeling",
        "xcopa": "language modeling",
        "xwinograd": "Coreference",
        "xnli": "NLI",
        "pawsx": "NLI",
    }
    return task_groups[task]


def final_eval_filter_conditions(df):
    return (
        valid_settings(df)
        & eval_settings("best", df)
        & eval_template_selection(df)
        & last_checkpoint(df)
        & num_few_shot_samples_all(df)
        & final_eval_splits(df)
    )


def multi_final_eval_filter_conditions(df):
    return (
        valid_settings(df)
        & eval_settings("best", df)
        & eval_template_selection(df)
        & multilingual_checkpoints(df)
        & last_checkpoint(df)
        & num_few_shot_samples(df)
        & multi_final_eval_splits(df)
        & all_multi_eval_tasks(df)
    )


def load_from_tsv_and_filter(in_tsv, filter_conditions):
    df = pd.read_csv(in_tsv, sep="\t", index_col=False).iloc[:, 1:]
    df = df.drop_duplicates()

    df["model"] = df.model_name.apply(lambda x: x.split("__step")[0])
    df["step"] = df.model_name.apply(lambda x: int(x.split("__step")[1]))
    df["model_size"] = df.model_name.apply(
        lambda x: float(x.split("_")[1][:-1])
        if x.split("_")[1].endswith("B")
        else float(x.split("_")[1][:-1]) / 1000
    )
    df["meta_task"] = df.task.apply(lambda x: x.split("__", 1)[0])
    df["model_id"] = df.model_name.apply(lambda x: x.split("__step", 1)[0])

    df = df[filter_conditions(df)]
    df["task_group"] = df.task.apply(lambda x: get_task_group(x))
    df["resource_level"] = df.language.apply(lambda x: get_resource_level(x))
    return df


def verify_multilingual_few_shot_learning_results(df, model_name=None):
    model_names = df["model_id"].unique() if model_name is None else [model_name]

    for m_name in model_names:
        print(f"Checking {m_name} predictions...")
        model_df = df[df["model_id"] == m_name]
        grouped_model_df = model_df.groupby(["task", "nb_few_shot_samples"])
        for group_key, _ in grouped_model_df:
            group = grouped_model_df.get_group(group_key)
            task, nb_few_shot_samples = group_key
            task_langs = multi_eval_tasks_langs[task]
            collected_task_langs = group["language"].to_list()

            print(group_key, collected_task_langs)
            if len(collected_task_langs) > len(task_langs):
                print("Warning: duplicated results collected")
            if len(collected_task_langs) < len(task_langs):
                print("Warning: missing language in result, expecting {task_langs}")
        print()


def verify_multilingual_big_run_paper_results():
    result_tables = {
        "multi-dense-564M": "/checkpoint/xianl/few_shot/dense_564M_lang30_new_cc100_xl_unigram_mutli_tasks_v1_test/results.tsv",
        "multi-dense-1.7B": "/checkpoint/xianl/few_shot/dense_1.7B_lang30_new_cc100_xl_unigram_mutli_tasks_v1_test/results.tsv",
        "multi-dense-7.5B": "/checkpoint/victorialin/few_shot/dense_7.5B_lang30_new_cc100_xl_unigram_mutli_tasks_v1/results.tsv",
    }

    def scaling_nb_few_shots(df):
        return (
            (df["nb_few_shot_samples"] == 0)
            | (df["nb_few_shot_samples"] == 1)
            | (df["nb_few_shot_samples"] == 4)
            | (df["nb_few_shot_samples"] == 32)
            | ((df["nb_few_shot_samples"] == 128) & (df["task"] != "storycloze"))
        )

    dfs = {}
    for key in result_tables:
        df = load_from_tsv_and_filter(
            result_tables[key], filter_conditions=multi_final_eval_filter_conditions
        )
        dfs[key] = df[scaling_nb_few_shots(df)]
    multi_result_df = pd.concat(dfs.values())

    verify_multilingual_few_shot_learning_results(multi_result_df)


if __name__ == "__main__":
    verify_multilingual_big_run_paper_results()
