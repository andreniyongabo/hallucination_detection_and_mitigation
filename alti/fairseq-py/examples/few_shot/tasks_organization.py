
import csv
from pathlib import Path
from examples.few_shot.tasks import get_all_tasks
from typing import *

DATA_DIR = Path(__file__).resolve().parent / "data"


def get_blimp_task_to_group_map():
    blimp_results = DATA_DIR / "blimp" / "blimp_full_results_summary.csv"

    blimp_task_to_group = {}
    with open(str(blimp_results)) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            task_name = row["UID"].lower()
            group = row["linguistics_term"]
            
            blimp_task = f"blimp__{task_name}"
            blimp_task_to_group[blimp_task] = f"blimp__{group}"

    return blimp_task_to_group


def get_blimp_task_groups():
    blimp_task_to_group = get_blimp_task_to_group_map()

    task_groups = {}
    for k,v in blimp_task_to_group.items():
        if v not in task_groups:
            task_groups[v] = []
        
        task_groups[v].append(k)
    
    return task_groups


def get_task_display_groups():
    # blimp
    task_display_groups = {
        "blimp__all": [x for x in get_all_tasks() if x.startswith("blimp__")],
        "all": [x for x in get_all_tasks()],
        "all_no_blimp": [x for x in get_all_tasks() if not x.startswith("blimp__")],
    }
    task_display_groups.update(get_blimp_task_groups())

    # science qa:
    task_display_groups.update({
        "scienceqa": ["arceasy", "arcchallenge", "openbookqa"],
        "super_glue": ["cb", "boolq", "copa", "wic", "wsc", "rte"],
        "commonsense": ["piqa", "winogrande", "hellaswag",],
        "LM": ["storycloze", "hellaswag"],
        "diagnosis_semantic": ['diagnosisbrand', 'diagnosiscity', 'diagnosiscountry', 'diagnosisname'],
        "diagnosis_positional": ['diagnosispos1', 'diagnosispos2', 'diagnosispos3', 'diagnosispos4'],
        "diagnosis_all": ['diagnosisbrand', 'diagnosiscity', 'diagnosiscountry', 'diagnosisname', 'diagnosispos1', 'diagnosispos2', 'diagnosispos3', 'diagnosispos4']
    })

    return task_display_groups


def get_tasks_to_groups_mapping(task_display_groups=None):
    """Returns tasks as keys and the grouping that they are part of as values.

    Args:
        task_display_groups ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if task_display_groups is None:
        task_display_groups = get_task_display_groups()
    task_to_groups = {}
    for tg, ts in task_display_groups.items():
        for tn in ts:
            if tn not in task_to_groups:
                task_to_groups[tn] = []
            task_to_groups[tn].append(tg)
    
    return task_to_groups


def invert_dict(dict_to_invert: Dict[str, List]):
    """Inverts a one-to-many dictionary:
    {"T1" : ["G1", "G2"]
     "T2" : ["G2"]}
     ->
    {"G1" : ["T1"]
     "G2" : ["T1", "T2"]}

    Args:
        dict_to_invert (Dict[str, List]): Dictionary with list values.

    Returns:
        [Dict[str, List]]: Inverted dict
    """
    inverted_dict = {}
    for k,v in dict_to_invert.items():
        for vv in v:
            if vv not in inverted_dict:
                inverted_dict[vv] = []
            inverted_dict[vv].append(k)
    inverted_dict = sorted([(k, sorted(list(set(v)))) for k,v in inverted_dict.items()], key = lambda k: k[0])
    inverted_dict = {k:v for k,v in inverted_dict}

    return inverted_dict

def get_groups_to_groups_mapping(task_display_groups=None):
    """This returns a mapping between group and 
    the groups that its tasks participates in.

    Returns:
        [dict]: Mapping between group and the groups that its tasks participates in.
    """
    if task_display_groups is None:
        task_display_groups = get_task_display_groups()
    task_to_groups = get_tasks_to_groups_mapping(task_display_groups)

    group_to_groups = {}
    for tg, ts in task_display_groups.items():
        for tn in ts:
            task_groups = [tgg for tgg in task_to_groups.get(tn, []) if tgg != tg]
            if tg not in group_to_groups:
                group_to_groups[tg] = []
            group_to_groups[tg].extend(task_groups)
    
    group_to_groups = {k:list(set(v)) for k,v in group_to_groups.items()}
    group_to_groups["blimp__all"] = ["all"]
    group_to_groups["all"] = []

    return group_to_groups


old_tasks_settings_default_eval_sets = {  # these old settings are used to update existing results and make compatable with the new collect_results.py
    "addition2digit": {"eval_set": "two_digit_addition", "train_set": "two_digit_addition", "valid_set": None},
    "addition3digit": {"eval_set": "three_digit_addition", "train_set": "three_digit_addition", "valid_set": None},
    "addition4digit": {"eval_set": "four_digit_addition", "train_set": "four_digit_addition", "valid_set": None},
    "addition5digit": {"eval_set": "five_digit_addition", "train_set": "five_digit_addition", "valid_set": None},
    "addition6digit": {"eval_set": "six_digit_addition", "train_set": "six_digit_addition", "valid_set": None},
    "anagrams1": {"eval_set": "mid_word_1_anagrams", "train_set": "mid_word_1_anagrams", "valid_set": None},
    "anagrams2": {"eval_set": "mid_word_2_anagrams", "train_set": "mid_word_2_anagrams", "valid_set": None},
    "anlir1": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "anlir2": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "anlir3": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "arcchallenge": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "arceasy": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "blimp__adjunct_island": {"eval_set": "adjunct_island", "train_set": "adjunct_island", "valid_set": None},
    "blimp__anaphor_gender_agreement": {"eval_set": "anaphor_gender_agreement", "train_set": "anaphor_gender_agreement", "valid_set": None},
    "blimp__anaphor_number_agreement": {"eval_set": "anaphor_number_agreement", "train_set": "anaphor_number_agreement", "valid_set": None},
    "blimp__animate_subject_passive": {"eval_set": "animate_subject_passive", "train_set": "animate_subject_passive", "valid_set": None},
    "blimp__animate_subject_trans": {"eval_set": "animate_subject_trans", "train_set": "animate_subject_trans", "valid_set": None},
    "blimp__causative": {"eval_set": "causative", "train_set": "causative", "valid_set": None},
    "blimp__complex_np_island": {"eval_set": "complex_np_island", "train_set": "complex_np_island", "valid_set": None},
    "blimp__coordinate_structure_constraint_complex_left_branch": {"eval_set": "coordinate_structure_constraint_complex_left_branch", "train_set": "coordinate_structure_constraint_complex_left_branch", "valid_set": None},
    "blimp__coordinate_structure_constraint_object_extraction": {"eval_set": "coordinate_structure_constraint_object_extraction", "train_set": "coordinate_structure_constraint_object_extraction", "valid_set": None},
    "blimp__determiner_noun_agreement_1": {"eval_set": "determiner_noun_agreement_1", "train_set": "determiner_noun_agreement_1", "valid_set": None},
    "blimp__determiner_noun_agreement_2": {"eval_set": "determiner_noun_agreement_2", "train_set": "determiner_noun_agreement_2", "valid_set": None},
    "blimp__determiner_noun_agreement_irregular_1": {"eval_set": "determiner_noun_agreement_irregular_1", "train_set": "determiner_noun_agreement_irregular_1", "valid_set": None},
    "blimp__determiner_noun_agreement_irregular_2": {"eval_set": "determiner_noun_agreement_irregular_2", "train_set": "determiner_noun_agreement_irregular_2", "valid_set": None},
    "blimp__determiner_noun_agreement_with_adj_2": {"eval_set": "determiner_noun_agreement_with_adj_2", "train_set": "determiner_noun_agreement_with_adj_2", "valid_set": None},
    "blimp__determiner_noun_agreement_with_adj_irregular_1": {"eval_set": "determiner_noun_agreement_with_adj_irregular_1", "train_set": "determiner_noun_agreement_with_adj_irregular_1", "valid_set": None},
    "blimp__determiner_noun_agreement_with_adj_irregular_2": {"eval_set": "determiner_noun_agreement_with_adj_irregular_2", "train_set": "determiner_noun_agreement_with_adj_irregular_2", "valid_set": None},
    "blimp__determiner_noun_agreement_with_adjective_1": {"eval_set": "determiner_noun_agreement_with_adjective_1", "train_set": "determiner_noun_agreement_with_adjective_1", "valid_set": None},
    "blimp__distractor_agreement_relational_noun": {"eval_set": "distractor_agreement_relational_noun", "train_set": "distractor_agreement_relational_noun", "valid_set": None},
    "blimp__distractor_agreement_relative_clause": {"eval_set": "distractor_agreement_relative_clause", "train_set": "distractor_agreement_relative_clause", "valid_set": None},
    "blimp__drop_argument": {"eval_set": "drop_argument", "train_set": "drop_argument", "valid_set": None},
    "blimp__ellipsis_n_bar_1": {"eval_set": "ellipsis_n_bar_1", "train_set": "ellipsis_n_bar_1", "valid_set": None},
    "blimp__ellipsis_n_bar_2": {"eval_set": "ellipsis_n_bar_2", "train_set": "ellipsis_n_bar_2", "valid_set": None},
    "blimp__existential_there_object_raising": {"eval_set": "existential_there_object_raising", "train_set": "existential_there_object_raising", "valid_set": None},
    "blimp__existential_there_quantifiers_1": {"eval_set": "existential_there_quantifiers_1", "train_set": "existential_there_quantifiers_1", "valid_set": None},
    "blimp__existential_there_quantifiers_2": {"eval_set": "existential_there_quantifiers_2", "train_set": "existential_there_quantifiers_2", "valid_set": None},
    "blimp__existential_there_subject_raising": {"eval_set": "existential_there_subject_raising", "train_set": "existential_there_subject_raising", "valid_set": None},
    "blimp__expletive_it_object_raising": {"eval_set": "expletive_it_object_raising", "train_set": "expletive_it_object_raising", "valid_set": None},
    "blimp__inchoative": {"eval_set": "inchoative", "train_set": "inchoative", "valid_set": None},
    "blimp__intransitive": {"eval_set": "intransitive", "train_set": "intransitive", "valid_set": None},
    "blimp__irregular_past_participle_adjectives": {"eval_set": "irregular_past_participle_adjectives", "train_set": "irregular_past_participle_adjectives", "valid_set": None},
    "blimp__irregular_past_participle_verbs": {"eval_set": "irregular_past_participle_verbs", "train_set": "irregular_past_participle_verbs", "valid_set": None},
    "blimp__irregular_plural_subject_verb_agreement_1": {"eval_set": "irregular_plural_subject_verb_agreement_1", "train_set": "irregular_plural_subject_verb_agreement_1", "valid_set": None},
    "blimp__irregular_plural_subject_verb_agreement_2": {"eval_set": "irregular_plural_subject_verb_agreement_2", "train_set": "irregular_plural_subject_verb_agreement_2", "valid_set": None},
    "blimp__left_branch_island_echo_question": {"eval_set": "left_branch_island_echo_question", "train_set": "left_branch_island_echo_question", "valid_set": None},
    "blimp__left_branch_island_simple_question": {"eval_set": "left_branch_island_simple_question", "train_set": "left_branch_island_simple_question", "valid_set": None},
    "blimp__matrix_question_npi_licensor_present": {"eval_set": "matrix_question_npi_licensor_present", "train_set": "matrix_question_npi_licensor_present", "valid_set": None},
    "blimp__npi_present_1": {"eval_set": "npi_present_1", "train_set": "npi_present_1", "valid_set": None},
    "blimp__npi_present_2": {"eval_set": "npi_present_2", "train_set": "npi_present_2", "valid_set": None},
    "blimp__only_npi_licensor_present": {"eval_set": "only_npi_licensor_present", "train_set": "only_npi_licensor_present", "valid_set": None},
    "blimp__only_npi_scope": {"eval_set": "only_npi_scope", "train_set": "only_npi_scope", "valid_set": None},
    "blimp__passive_1": {"eval_set": "passive_1", "train_set": "passive_1", "valid_set": None},
    "blimp__passive_2": {"eval_set": "passive_2", "train_set": "passive_2", "valid_set": None},
    "blimp__principle_a_c_command": {"eval_set": "principle_a_c_command", "train_set": "principle_a_c_command", "valid_set": None},
    "blimp__principle_a_case_1": {"eval_set": "principle_a_case_1", "train_set": "principle_a_case_1", "valid_set": None},
    "blimp__principle_a_case_2": {"eval_set": "principle_a_case_2", "train_set": "principle_a_case_2", "valid_set": None},
    "blimp__principle_a_domain_1": {"eval_set": "principle_a_domain_1", "train_set": "principle_a_domain_1", "valid_set": None},
    "blimp__principle_a_domain_2": {"eval_set": "principle_a_domain_2", "train_set": "principle_a_domain_2", "valid_set": None},
    "blimp__principle_a_domain_3": {"eval_set": "principle_a_domain_3", "train_set": "principle_a_domain_3", "valid_set": None},
    "blimp__principle_a_reconstruction": {"eval_set": "principle_a_reconstruction", "train_set": "principle_a_reconstruction", "valid_set": None},
    "blimp__regular_plural_subject_verb_agreement_1": {"eval_set": "regular_plural_subject_verb_agreement_1", "train_set": "regular_plural_subject_verb_agreement_1", "valid_set": None},
    "blimp__regular_plural_subject_verb_agreement_2": {"eval_set": "regular_plural_subject_verb_agreement_2", "train_set": "regular_plural_subject_verb_agreement_2", "valid_set": None},
    "blimp__sentential_negation_npi_licensor_present": {"eval_set": "sentential_negation_npi_licensor_present", "train_set": "sentential_negation_npi_licensor_present", "valid_set": None},
    "blimp__sentential_negation_npi_scope": {"eval_set": "sentential_negation_npi_scope", "train_set": "sentential_negation_npi_scope", "valid_set": None},
    "blimp__sentential_subject_island": {"eval_set": "sentential_subject_island", "train_set": "sentential_subject_island", "valid_set": None},
    "blimp__superlative_quantifiers_1": {"eval_set": "superlative_quantifiers_1", "train_set": "superlative_quantifiers_1", "valid_set": None},
    "blimp__superlative_quantifiers_2": {"eval_set": "superlative_quantifiers_2", "train_set": "superlative_quantifiers_2", "valid_set": None},
    "blimp__tough_vs_raising_1": {"eval_set": "tough_vs_raising_1", "train_set": "tough_vs_raising_1", "valid_set": None},
    "blimp__tough_vs_raising_2": {"eval_set": "tough_vs_raising_2", "train_set": "tough_vs_raising_2", "valid_set": None},
    "blimp__transitive": {"eval_set": "transitive", "train_set": "transitive", "valid_set": None},
    "blimp__wh_island": {"eval_set": "wh_island", "train_set": "wh_island", "valid_set": None},
    "blimp__wh_questions_object_gap": {"eval_set": "wh_questions_object_gap", "train_set": "wh_questions_object_gap", "valid_set": None},
    "blimp__wh_questions_subject_gap": {"eval_set": "wh_questions_subject_gap", "train_set": "wh_questions_subject_gap", "valid_set": None},
    "blimp__wh_questions_subject_gap_long_distance": {"eval_set": "wh_questions_subject_gap_long_distance", "train_set": "wh_questions_subject_gap_long_distance", "valid_set": None},
    "blimp__wh_vs_that_no_gap": {"eval_set": "wh_vs_that_no_gap", "train_set": "wh_vs_that_no_gap", "valid_set": None},
    "blimp__wh_vs_that_no_gap_long_distance": {"eval_set": "wh_vs_that_no_gap_long_distance", "train_set": "wh_vs_that_no_gap_long_distance", "valid_set": None},
    "blimp__wh_vs_that_with_gap": {"eval_set": "wh_vs_that_with_gap", "train_set": "wh_vs_that_with_gap", "valid_set": None},
    "blimp__wh_vs_that_with_gap_long_distance": {"eval_set": "wh_vs_that_with_gap_long_distance", "train_set": "wh_vs_that_with_gap_long_distance", "valid_set": None},
    "boolq": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "cb": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "commonsenseqa": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "copa": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "crowspairs": {"eval_set": "test", "train_set": None, "valid_set": None},
    "cycledletters": {"eval_set": "cycle_letters_in_word", "train_set": "cycle_letters_in_word", "valid_set": None},
    "ethoszeroshot": {"eval_set": "zero_shot", "train_set": None, "valid_set": None},
    "exams": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "hellaswag": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "lama_conceptnet": {"eval_set": "conceptnet_test", "train_set": None, "valid_set": None},
    "lama_googlere": {"eval_set": "googlere", "train_set": None, "valid_set": None},
    "lama_squad": {"eval_set": "squad_test", "train_set": None, "valid_set": None},
    "lama_trex": {"eval_set": "trex", "train_set": None, "valid_set": None},
    "mlama_googlere": {"eval_set": "googlere", "train_set": None, "valid_set": None},
    "mlama_trex": {"eval_set": "trex", "train_set": None, "valid_set": None},
    "mnlimatched": {"eval_set": "dev_matched", "train_set": "train", "valid_set": None},
    "mnlimismatched": {"eval_set": "dev_mismatched", "train_set": "train", "valid_set": None},
    "multiplication2digit": {"eval_set": "single_digit_three_ops", "train_set": "single_digit_three_ops", "valid_set": None},
    "multirc": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "naturalinstructions__subtask001_quoref_question_generation": {"eval_set": "subtask001_quoref_question_generation", "train_set": "subtask001_quoref_question_generation", "valid_set": None},
    "naturalinstructions__subtask002_quoref_answer_generation": {"eval_set": "subtask002_quoref_answer_generation", "train_set": "subtask002_quoref_answer_generation", "valid_set": None},
    "naturalinstructions__subtask003_mctaco_question_generation_event_duration": {"eval_set": "subtask003_mctaco_question_generation_event_duration", "train_set": "subtask003_mctaco_question_generation_event_duration", "valid_set": None},
    "naturalinstructions__subtask004_mctaco_answer_generation_event_duration": {"eval_set": "subtask004_mctaco_answer_generation_event_duration", "train_set": "subtask004_mctaco_answer_generation_event_duration", "valid_set": None},
    "naturalinstructions__subtask005_mctaco_wrong_answer_generation_event_duration": {"eval_set": "subtask005_mctaco_wrong_answer_generation_event_duration", "train_set": "subtask005_mctaco_wrong_answer_generation_event_duration", "valid_set": None},
    "naturalinstructions__subtask006_mctaco_question_generation_transient_stationary": {"eval_set": "subtask006_mctaco_question_generation_transient_stationary", "train_set": "subtask006_mctaco_question_generation_transient_stationary", "valid_set": None},
    "naturalinstructions__subtask007_mctaco_answer_generation_transient_stationary": {"eval_set": "subtask007_mctaco_answer_generation_transient_stationary", "train_set": "subtask007_mctaco_answer_generation_transient_stationary", "valid_set": None},
    "naturalinstructions__subtask008_mctaco_wrong_answer_generation_transient_stationary": {"eval_set": "subtask008_mctaco_wrong_answer_generation_transient_stationary", "train_set": "subtask008_mctaco_wrong_answer_generation_transient_stationary", "valid_set": None},
    "naturalinstructions__subtask009_mctaco_question_generation_event_ordering": {"eval_set": "subtask009_mctaco_question_generation_event_ordering", "train_set": "subtask009_mctaco_question_generation_event_ordering", "valid_set": None},
    "naturalinstructions__subtask010_mctaco_answer_generation_event_ordering": {"eval_set": "subtask010_mctaco_answer_generation_event_ordering", "train_set": "subtask010_mctaco_answer_generation_event_ordering", "valid_set": None},
    "naturalinstructions__subtask011_mctaco_wrong_answer_generation_event_ordering": {"eval_set": "subtask011_mctaco_wrong_answer_generation_event_ordering", "train_set": "subtask011_mctaco_wrong_answer_generation_event_ordering", "valid_set": None},
    "naturalinstructions__subtask012_mctaco_question_generation_absolute_timepoint": {"eval_set": "subtask012_mctaco_question_generation_absolute_timepoint", "train_set": "subtask012_mctaco_question_generation_absolute_timepoint", "valid_set": None},
    "naturalinstructions__subtask013_mctaco_answer_generation_absolute_timepoint": {"eval_set": "subtask013_mctaco_answer_generation_absolute_timepoint", "train_set": "subtask013_mctaco_answer_generation_absolute_timepoint", "valid_set": None},
    "naturalinstructions__subtask014_mctaco_wrong_answer_generation_absolute_timepoint": {"eval_set": "subtask014_mctaco_wrong_answer_generation_absolute_timepoint", "train_set": "subtask014_mctaco_wrong_answer_generation_absolute_timepoint", "valid_set": None},
    "naturalinstructions__subtask015_mctaco_question_generation_frequency": {"eval_set": "subtask015_mctaco_question_generation_frequency", "train_set": "subtask015_mctaco_question_generation_frequency", "valid_set": None},
    "naturalinstructions__subtask016_mctaco_answer_generation_frequency": {"eval_set": "subtask016_mctaco_answer_generation_frequency", "train_set": "subtask016_mctaco_answer_generation_frequency", "valid_set": None},
    "naturalinstructions__subtask017_mctaco_wrong_answer_generation_frequency": {"eval_set": "subtask017_mctaco_wrong_answer_generation_frequency", "train_set": "subtask017_mctaco_wrong_answer_generation_frequency", "valid_set": None},
    "naturalinstructions__subtask018_mctaco_temporal_reasoning_presence": {"eval_set": "subtask018_mctaco_temporal_reasoning_presence", "train_set": "subtask018_mctaco_temporal_reasoning_presence", "valid_set": None},
    "naturalinstructions__subtask019_mctaco_temporal_reasoning_category": {"eval_set": "subtask019_mctaco_temporal_reasoning_category", "train_set": "subtask019_mctaco_temporal_reasoning_category", "valid_set": None},
    "naturalinstructions__subtask020_mctaco_span_based_question": {"eval_set": "subtask020_mctaco_span_based_question", "train_set": "subtask020_mctaco_span_based_question", "valid_set": None},
    "naturalinstructions__subtask021_mctaco_grammatical_logical": {"eval_set": "subtask021_mctaco_grammatical_logical", "train_set": "subtask021_mctaco_grammatical_logical", "valid_set": None},
    "naturalinstructions__subtask022_cosmosqa_passage_inappropriate_binary": {"eval_set": "subtask022_cosmosqa_passage_inappropriate_binary", "train_set": "subtask022_cosmosqa_passage_inappropriate_binary", "valid_set": None},
    "naturalinstructions__subtask023_cosmosqa_question_generation": {"eval_set": "subtask023_cosmosqa_question_generation", "train_set": "subtask023_cosmosqa_question_generation", "valid_set": None},
    "naturalinstructions__subtask024_cosmosqa_answer_generation": {"eval_set": "subtask024_cosmosqa_answer_generation", "train_set": "subtask024_cosmosqa_answer_generation", "valid_set": None},
    "naturalinstructions__subtask025_cosmosqa_incorrect_answer_generation": {"eval_set": "subtask025_cosmosqa_incorrect_answer_generation", "train_set": "subtask025_cosmosqa_incorrect_answer_generation", "valid_set": None},
    "naturalinstructions__subtask026_drop_question_generation": {"eval_set": "subtask026_drop_question_generation", "train_set": "subtask026_drop_question_generation", "valid_set": None},
    "naturalinstructions__subtask027_drop_answer_type_generation": {"eval_set": "subtask027_drop_answer_type_generation", "train_set": "subtask027_drop_answer_type_generation", "valid_set": None},
    "naturalinstructions__subtask028_drop_answer_generation": {"eval_set": "subtask028_drop_answer_generation", "train_set": "subtask028_drop_answer_generation", "valid_set": None},
    "naturalinstructions__subtask029_winogrande_full_object": {"eval_set": "subtask029_winogrande_full_object", "train_set": "subtask029_winogrande_full_object", "valid_set": None},
    "naturalinstructions__subtask030_winogrande_full_person": {"eval_set": "subtask030_winogrande_full_person", "train_set": "subtask030_winogrande_full_person", "valid_set": None},
    "naturalinstructions__subtask031_winogrande_question_generation_object": {"eval_set": "subtask031_winogrande_question_generation_object", "train_set": "subtask031_winogrande_question_generation_object", "valid_set": None},
    "naturalinstructions__subtask032_winogrande_question_generation_person": {"eval_set": "subtask032_winogrande_question_generation_person", "train_set": "subtask032_winogrande_question_generation_person", "valid_set": None},
    "naturalinstructions__subtask033_winogrande_answer_generation": {"eval_set": "subtask033_winogrande_answer_generation", "train_set": "subtask033_winogrande_answer_generation", "valid_set": None},
    "naturalinstructions__subtask034_winogrande_question_modification_object": {"eval_set": "subtask034_winogrande_question_modification_object", "train_set": "subtask034_winogrande_question_modification_object", "valid_set": None},
    "naturalinstructions__subtask035_winogrande_question_modification_person": {"eval_set": "subtask035_winogrande_question_modification_person", "train_set": "subtask035_winogrande_question_modification_person", "valid_set": None},
    "naturalinstructions__subtask036_qasc_topic_word_to_generate_related_fact": {"eval_set": "subtask036_qasc_topic_word_to_generate_related_fact", "train_set": "subtask036_qasc_topic_word_to_generate_related_fact", "valid_set": None},
    "naturalinstructions__subtask037_qasc_generate_related_fact": {"eval_set": "subtask037_qasc_generate_related_fact", "train_set": "subtask037_qasc_generate_related_fact", "valid_set": None},
    "naturalinstructions__subtask038_qasc_combined_fact": {"eval_set": "subtask038_qasc_combined_fact", "train_set": "subtask038_qasc_combined_fact", "valid_set": None},
    "naturalinstructions__subtask039_qasc_find_overlapping_words": {"eval_set": "subtask039_qasc_find_overlapping_words", "train_set": "subtask039_qasc_find_overlapping_words", "valid_set": None},
    "naturalinstructions__subtask040_qasc_question_generation": {"eval_set": "subtask040_qasc_question_generation", "train_set": "subtask040_qasc_question_generation", "valid_set": None},
    "naturalinstructions__subtask041_qasc_answer_generation": {"eval_set": "subtask041_qasc_answer_generation", "train_set": "subtask041_qasc_answer_generation", "valid_set": None},
    "naturalinstructions__subtask042_qasc_incorrect_option_generation": {"eval_set": "subtask042_qasc_incorrect_option_generation", "train_set": "subtask042_qasc_incorrect_option_generation", "valid_set": None},
    "naturalinstructions__subtask043_essential_terms_answering_incomplete_questions": {"eval_set": "subtask043_essential_terms_answering_incomplete_questions", "train_set": "subtask043_essential_terms_answering_incomplete_questions", "valid_set": None},
    "naturalinstructions__subtask044_essential_terms_identifying_essential_words": {"eval_set": "subtask044_essential_terms_identifying_essential_words", "train_set": "subtask044_essential_terms_identifying_essential_words", "valid_set": None},
    "naturalinstructions__subtask045_miscellaneous_sentence_paraphrasing": {"eval_set": "subtask045_miscellaneous_sentence_paraphrasing", "train_set": "subtask045_miscellaneous_sentence_paraphrasing", "valid_set": None},
    "naturalinstructions__subtask046_miscellaenous_question_typing": {"eval_set": "subtask046_miscellaenous_question_typing", "train_set": "subtask046_miscellaenous_question_typing", "valid_set": None},
    "naturalinstructions__subtask047_misc_answering_science_questions": {"eval_set": "subtask047_misc_answering_science_questions", "train_set": "subtask047_misc_answering_science_questions", "valid_set": None},
    "naturalinstructions__subtask048_multirc_question_generation": {"eval_set": "subtask048_multirc_question_generation", "train_set": "subtask048_multirc_question_generation", "valid_set": None},
    "naturalinstructions__subtask049_multirc_questions_needed_to_answer": {"eval_set": "subtask049_multirc_questions_needed_to_answer", "train_set": "subtask049_multirc_questions_needed_to_answer", "valid_set": None},
    "naturalinstructions__subtask050_multirc_answerability": {"eval_set": "subtask050_multirc_answerability", "train_set": "subtask050_multirc_answerability", "valid_set": None},
    "naturalinstructions__subtask051_multirc_correct_answer_single_sentence": {"eval_set": "subtask051_multirc_correct_answer_single_sentence", "train_set": "subtask051_multirc_correct_answer_single_sentence", "valid_set": None},
    "naturalinstructions__subtask052_multirc_identify_bad_question": {"eval_set": "subtask052_multirc_identify_bad_question", "train_set": "subtask052_multirc_identify_bad_question", "valid_set": None},
    "naturalinstructions__subtask053_multirc_correct_bad_question": {"eval_set": "subtask053_multirc_correct_bad_question", "train_set": "subtask053_multirc_correct_bad_question", "valid_set": None},
    "naturalinstructions__subtask054_multirc_write_correct_answer": {"eval_set": "subtask054_multirc_write_correct_answer", "train_set": "subtask054_multirc_write_correct_answer", "valid_set": None},
    "naturalinstructions__subtask055_multirc_write_incorrect_answer": {"eval_set": "subtask055_multirc_write_incorrect_answer", "train_set": "subtask055_multirc_write_incorrect_answer", "valid_set": None},
    "naturalinstructions__subtask056_multirc_classify_correct_answer": {"eval_set": "subtask056_multirc_classify_correct_answer", "train_set": "subtask056_multirc_classify_correct_answer", "valid_set": None},
    "naturalinstructions__subtask057_multirc_classify_incorrect_answer": {"eval_set": "subtask057_multirc_classify_incorrect_answer", "train_set": "subtask057_multirc_classify_incorrect_answer", "valid_set": None},
    "naturalinstructions__subtask058_multirc_question_answering": {"eval_set": "subtask058_multirc_question_answering", "train_set": "subtask058_multirc_question_answering", "valid_set": None},
    "naturalinstructions__subtask059_ropes_story_generation": {"eval_set": "subtask059_ropes_story_generation", "train_set": "subtask059_ropes_story_generation", "valid_set": None},
    "naturalinstructions__subtask060_ropes_question_generation": {"eval_set": "subtask060_ropes_question_generation", "train_set": "subtask060_ropes_question_generation", "valid_set": None},
    "naturalinstructions__subtask061_ropes_answer_generation": {"eval_set": "subtask061_ropes_answer_generation", "train_set": "subtask061_ropes_answer_generation", "valid_set": None},
    "naturalquestions": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "openbookqa": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "pawsx": {"eval_set": "dev", "train_set": "test", "valid_set": None},
    "piqa": {"eval_set": "valid", "train_set": "train", "valid_set": None},
    "processtext": {"eval_set": None, "train_set": None, "valid_set": None},
    "realtoxicityprompts": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "record": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "reversedwords": {"eval_set": "reversed_words", "train_set": "reversed_words", "valid_set": None},
    "rte": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "satanalogies": {"eval_set": "test", "train_set": "test", "valid_set": None},
    "simplification": {"eval_set": "valid", "train_set": "test", "valid_set": None},
    "singledigit3ops": {"eval_set": "single_digit_three_ops", "train_set": "single_digit_three_ops", "valid_set": None},
    "snli": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "stereoset": {"eval_set": "dev", "train_set": None, "valid_set": None},
    "storycloze": {"eval_set": "test2016", "train_set": "val2016", "valid_set": None},
    "subtraction2digit": {"eval_set": "two_digit_subtraction", "train_set": "two_digit_subtraction", "valid_set": None},
    "subtraction3digit": {"eval_set": "three_digit_subtraction", "train_set": "three_digit_subtraction", "valid_set": None},
    "subtraction4digit": {"eval_set": "four_digit_subtraction", "train_set": "four_digit_subtraction", "valid_set": None},
    "subtraction5digit": {"eval_set": "five_digit_subtraction", "train_set": "five_digit_subtraction", "valid_set": None},
    "subtraction6digit": {"eval_set": "six_digit_subtraction", "train_set": "six_digit_subtraction", "valid_set": None},
    "sumofdigits": {"eval_set": "sum_of_digits", "train_set": "sum_of_digits", "valid_set": None},
    "symbolinsertion": {"eval_set": "random_insertion_in_word", "train_set": "random_insertion_in_word", "valid_set": None},
    "triviaqa": {"eval_set": "dev", "train_set": "train", "valid_set": None},
    "webquestions": {"eval_set": "test", "train_set": "train", "valid_set": None},
    "wic": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "winograd": {"eval_set": "test", "train_set": "test", "valid_set": None},
    "winogrande": {"eval_set": "dev", "train_set": "train_xl", "valid_set": None},
    "wmt14enfr": {"eval_set": "wmt14/full", "train_set": "wmt13", "valid_set": None},
    "wmt14fren": {"eval_set": "wmt14/full", "train_set": "wmt13", "valid_set": None},
    "wmt16deen": {"eval_set": "wmt16", "train_set": "wmt15", "valid_set": None},
    "wmt16ende": {"eval_set": "wmt16", "train_set": "wmt15", "valid_set": None},
    "wmt16enro": {"eval_set": "wmt16", "train_set": "wmt16", "valid_set": None},
    "wmt16roen": {"eval_set": "wmt16", "train_set": "wmt16", "valid_set": None},
    "wsc": {"eval_set": "val", "train_set": "train", "valid_set": None},
    "xcopa": {"eval_set": "val", "train_set": "test", "valid_set": None},
    "xnli": {"eval_set": "dev", "train_set": "test", "valid_set": None},
}

if __name__ == "__main__":
    # tasks = set(get_all_tasks())
    # for k,v in get_blimp_task_to_group_map().items():
    #     assert k in tasks, f"{k} not in tasks"

    #print(get_task_display_groups().keys())
    #print(invert_dict(get_groups_to_groups_mapping()))

    print("Old default configurations")
    print("task_name | train_set | eval_set | valid_set")
    for task_name in get_all_tasks():
        setting = old_tasks_settings_default_eval_sets[task_name]
        print(f"{task_name} | {setting['train_set']} | {setting['eval_set']} | {setting['valid_set']}")
    