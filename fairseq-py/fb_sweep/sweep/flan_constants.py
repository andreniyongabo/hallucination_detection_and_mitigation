flan_clusters = {
    "nli": [
        "flan__anli_r1_10templates",
        "flan__anli_r2_10templates",
        "flan__anli_r3_10templates",
        "flan__snli_10templates",
        "flan__qnli_10templates",
        "flan__wnli_10templates",
        "flan__mnli_matched_10templates",
        "flan__mnli_mismatched_10templates",
        "flan__cb_10templates",
        "flan__rte_10templates",
    ],
    "commonsense": [
        "flan__copa_10templates",
        "flan__hellaswag_10templates",
        "flan__piqa_10templates",
        "flan__story_cloze_10templates",
    ],
    "sentiment": [
        "flan__imdb_reviews_10templates",
        "flan__sentiment140_10templates",
        "flan__sst2_10templates",
        "flan__yelp_polarity_reviews_10templates",
    ],
    "paraphrase": [
        "flan__glue_mrpc_10templates",
        "flan__glue_qqp_10templates",
        "flan__paws_wiki_10templates",
        "flan__stsb_10templates",
    ],
    "qa": [
        "flan__arc_challenge_10templates",
        "flan__arc_easy_10templates",
        "flan__natural_questions_10templates",
        "flan__trivia_qa_10templates",
    ],
    "struct2text": [
        "flan__common_gen_10templates",
        "flan__dart_10templates",
        "flan__e2e_nlg_10templates",
        "flan__web_nlg_en_10templates",
    ],
    "mrc": [
        "flan__bool_q_10templates",
        "flan__drop_10templates",
        "flan__openbookqa_10templates",
        "flan__squad_v1_10templates",
        "flan__squad_v2_10templates",
        "flan__multirc_10templates",
    ],
    "mrc_with_commonsense": ["flan__cosmos_qa_10templates", "flan__record_10templates"],
    "coref": [
        "flan__winogrande_10templates",
        "flan__wsc_10templates"
        # DPR is missing
    ],
    "misc": [
        "flan__coqa_10templates",
        "flan__quac_10templates",
        "flan__trec_10templates",
        "flan__cola_10templates",
        "flan__wic_10templates",
        "flan__math_dataset_10templates",
        "flan__fix_punct_10templates",
    ],
    "summarization": [
        "flan__aeslc_10templates",
        "flan__ag_news_subset_10templates",
        "flan__cnn_dailymail_10templates",
        "flan__gigaword_10templates",
        # 'multi_news_10templates', # Has too many out of length samples
        "flan__newsroom_10templates",
        "flan__opinion_abstracts_idebate_10templates",
        "flan__opinion_abstracts_rotten_tomatoes_10templates",
        "flan__samsum_10templates",
        "flan__xsum_10templates",
        "flan__wiki_lingua_english_en_10templates",
    ],
    "translation": [
        "flan__para_crawl_enes_10templates",
        "flan__wmt16_translate_deen_10templates",
        "flan__wmt16_translate_fien_10templates",
        "flan__wmt16_translate_roen_10templates",
        "flan__wmt16_translate_ruen_10templates",
        "flan__wmt16_translate_tren_10templates"
        # 3 translation datasets missing
    ],
}
