import unittest

try:
    from examples.few_shot.tasks import get_all_tasks, init_task_with_custom_args
except FileNotFoundError:
    raise unittest.SkipTest(
        "Unable to import tasks module. Skipping all tests in test_few_shot_tasks.py"
    )


class TestFewShotTasks(unittest.TestCase):
    def test_if_all_tasks_are_available(self):
        """
        Some changes (such as removing abstract properties)
        can prevent tasks from being registered.
        In order to catch this early, we will validate that the old tasks are still available.
        """
        # fmt:off
        old_tasks = ['stereoset', 'crowspairs', 'ethoszeroshot', 'copa', 'pawsx', 'xcopa', 'hellaswag', 'storycloze', 'winograd', 'winogrande', 'piqa', 'arcchallenge', 'arceasy', 'openbookqa', 'commonsenseqa', 'exams', 'naturalquestions', 'triviaqa', 'webquestions', 'wic', 'boolq', 'cb', 'rte', 'wsc', 'record', 'multirc', 'snli', 'mnlimatched', 'mnlimismatched', 'anlir1', 'anlir2', 'anlir3', 'xnli', 'addition2digit', 'addition3digit', 'addition4digit', 'addition5digit', 'addition6digit', 'subtraction2digit', 'subtraction3digit', 'subtraction4digit', 'subtraction5digit', 'subtraction6digit', 'multiplication2digit', 'singledigit3ops', 'sumofdigits', 'cycledletters', 'anagrams1', 'anagrams2', 'symbolinsertion', 'reversedwords', 'wmt14fren', 'wmt14enfr', 'wmt16deen', 'wmt16ende', 'wmt16roen', 'wmt16enro', 'satanalogies', 'simplification', 'realtoxicityprompts', 'naturalinstructions__subtask001_quoref_question_generation', 'naturalinstructions__subtask002_quoref_answer_generation', 'naturalinstructions__subtask003_mctaco_question_generation_event_duration', 'naturalinstructions__subtask004_mctaco_answer_generation_event_duration', 'naturalinstructions__subtask005_mctaco_wrong_answer_generation_event_duration', 'naturalinstructions__subtask006_mctaco_question_generation_transient_stationary', 'naturalinstructions__subtask007_mctaco_answer_generation_transient_stationary', 'naturalinstructions__subtask008_mctaco_wrong_answer_generation_transient_stationary', 'naturalinstructions__subtask009_mctaco_question_generation_event_ordering', 'naturalinstructions__subtask010_mctaco_answer_generation_event_ordering', 'naturalinstructions__subtask011_mctaco_wrong_answer_generation_event_ordering', 'naturalinstructions__subtask012_mctaco_question_generation_absolute_timepoint', 'naturalinstructions__subtask013_mctaco_answer_generation_absolute_timepoint', 'naturalinstructions__subtask014_mctaco_wrong_answer_generation_absolute_timepoint', 'naturalinstructions__subtask015_mctaco_question_generation_frequency', 'naturalinstructions__subtask016_mctaco_answer_generation_frequency', 'naturalinstructions__subtask017_mctaco_wrong_answer_generation_frequency', 'naturalinstructions__subtask018_mctaco_temporal_reasoning_presence', 'naturalinstructions__subtask019_mctaco_temporal_reasoning_category', 'naturalinstructions__subtask020_mctaco_span_based_question', 'naturalinstructions__subtask021_mctaco_grammatical_logical', 'naturalinstructions__subtask022_cosmosqa_passage_inappropriate_binary', 'naturalinstructions__subtask023_cosmosqa_question_generation', 'naturalinstructions__subtask024_cosmosqa_answer_generation', 'naturalinstructions__subtask025_cosmosqa_incorrect_answer_generation', 'naturalinstructions__subtask026_drop_question_generation', 'naturalinstructions__subtask027_drop_answer_type_generation', 'naturalinstructions__subtask028_drop_answer_generation', 'naturalinstructions__subtask029_winogrande_full_object', 'naturalinstructions__subtask030_winogrande_full_person', 'naturalinstructions__subtask031_winogrande_question_generation_object', 'naturalinstructions__subtask032_winogrande_question_generation_person', 'naturalinstructions__subtask033_winogrande_answer_generation', 'naturalinstructions__subtask034_winogrande_question_modification_object', 'naturalinstructions__subtask035_winogrande_question_modification_person', 'naturalinstructions__subtask036_qasc_topic_word_to_generate_related_fact', 'naturalinstructions__subtask037_qasc_generate_related_fact', 'naturalinstructions__subtask038_qasc_combined_fact', 'naturalinstructions__subtask039_qasc_find_overlapping_words', 'naturalinstructions__subtask040_qasc_question_generation', 'naturalinstructions__subtask041_qasc_answer_generation', 'naturalinstructions__subtask042_qasc_incorrect_option_generation', 'naturalinstructions__subtask043_essential_terms_answering_incomplete_questions', 'naturalinstructions__subtask044_essential_terms_identifying_essential_words', 'naturalinstructions__subtask045_miscellaneous_sentence_paraphrasing', 'naturalinstructions__subtask046_miscellaenous_question_typing', 'naturalinstructions__subtask047_misc_answering_science_questions', 'naturalinstructions__subtask048_multirc_question_generation', 'naturalinstructions__subtask049_multirc_questions_needed_to_answer', 'naturalinstructions__subtask050_multirc_answerability', 'naturalinstructions__subtask051_multirc_correct_answer_single_sentence', 'naturalinstructions__subtask052_multirc_identify_bad_question', 'naturalinstructions__subtask053_multirc_correct_bad_question', 'naturalinstructions__subtask054_multirc_write_correct_answer', 'naturalinstructions__subtask055_multirc_write_incorrect_answer', 'naturalinstructions__subtask056_multirc_classify_correct_answer', 'naturalinstructions__subtask057_multirc_classify_incorrect_answer', 'naturalinstructions__subtask058_multirc_question_answering', 'naturalinstructions__subtask059_ropes_story_generation', 'naturalinstructions__subtask060_ropes_question_generation', 'naturalinstructions__subtask061_ropes_answer_generation', 'processtext', 'lama_conceptnet', 'lama_squad', 'lama_googlere', 'lama_trex', 'mlama_trex', 'mlama_googlere']  # noqa
        # fmt:on
        curr_tasks = set(get_all_tasks())
        for old_task in old_tasks:
            assert old_task in curr_tasks, f"{old_task} is missing!"

    def test_if_all_tasks_have_default_train_and_test_sets(self):
        curr_tasks = list(get_all_tasks())

        task_field_exceptions_set = {
            (
                "crowspairs",
                "train_set",
            ),  # RAI diagnostic task and only supports zero-shot
            ("crowspairs", "train_lang"),
            (
                "stereoset",
                "train_set",
            ),  # RAI diagnostic task and only supports zero-shot
            ("stereoset", "train_lang"),
            (
                "ethoszeroshot",
                "train_set",
            ),  # This task supports only zero-shot ethos few-shot will be implemented later
            ("ethoszeroshot", "train_lang"),
            (
                "lama_conceptnet",
                "train_set",
            ),  # LAMA is designed for zer-shot knowledge eval
            ("lama_conceptnet", "train_lang"),
            ("lama_googlere", "train_set"),
            ("lama_googlere", "train_lang"),
            ("lama_squad", "train_set"),
            ("lama_squad", "train_lang"),
            ("lama_trex", "train_set"),
            ("lama_trex", "train_lang"),
            (
                "mlama_googlere",
                "train_set",
            ),  # mlama tasks are designed fo zer-shot knowledge eval
            ("mlama_googlere", "train_lang"),
            ("mlama_trex", "train_set"),
            ("mlama_trex", "train_lang"),
            (
                "processtext",
                "eval_set",
            ),  # This task is used for debugging and eval_set is supposed to be any text file!
            ("processtext", "train_lang"),
            ("processtext", "train_set"),
            ("processtext", "train_lang"),
            ("gluediag", "train_set"),
            ("gluediag", "train_lang"),
            ("rai_pii_leaks", "train_set"),
            ("rai_pii_leaks", "train_lang"),
        }
        for task_name in curr_tasks:
            task_instance, task_args = init_task_with_custom_args(task_name)
            task_info = task_instance.eval_attributes()

            for attr_name in ["eval_set", "train_set", "language", "train_lang"]:
                assert (
                    attr_name in task_info
                ), f"{task_name} - {attr_name} not in eval_attributes"

                assert (
                    task_info[attr_name] is not None
                    or (task_name, attr_name) in task_field_exceptions_set
                ), f"{task_name} - {attr_name} is None!"


if __name__ == "__main__":
    unittest.main()
