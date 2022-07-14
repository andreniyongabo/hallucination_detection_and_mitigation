# fmt: off
import argparse 
import sys
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *
from examples.few_shot.tasks import get_tasks_by_group

if __name__ == "__main__":
    """
    Run random baselines for multi-choie tasks.
    
    Selected tasks: 
        - All multichoice tasks
    
    Setting:
        - Smallest model -- dummy run
        
    Commands:
        - Run locally
            # debug
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_baselines_random.py -t copa --local --dry-run
            
            # run all
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_baselines_random.py -t multichoice_tasks --local

    """

    # task settings
    available_tasks_settings = default_tasks_settings.copy()
    available_tasks_settings.update({
        "blimp_all": ("blimp_all", get_tasks_by_group("blimp"), {}),
    })
    task_run_groups = default_task_run_groups.copy()
    task_run_groups.update({
        "multichoice_tasks": [
            #'addition2digit', 'addition3digit', 'addition4digit', 'addition5digit', 'addition6digit', 'anagrams1', 'anagrams2', 
            'exams',
            'anlir1', 'anlir2', 'anlir3', 'arcchallenge', 'arceasy', 'blimp__adjunct_island', 'blimp__anaphor_gender_agreement', 'blimp__anaphor_number_agreement', 'blimp__animate_subject_passive', 'blimp__animate_subject_trans', 'blimp__causative', 'blimp__complex_np_island', 'blimp__coordinate_structure_constraint_complex_left_branch', 'blimp__coordinate_structure_constraint_object_extraction', 'blimp__determiner_noun_agreement_1', 'blimp__determiner_noun_agreement_2', 'blimp__determiner_noun_agreement_irregular_1', 'blimp__determiner_noun_agreement_irregular_2', 'blimp__determiner_noun_agreement_with_adj_2', 'blimp__determiner_noun_agreement_with_adj_irregular_1', 'blimp__determiner_noun_agreement_with_adj_irregular_2', 'blimp__determiner_noun_agreement_with_adjective_1', 'blimp__distractor_agreement_relational_noun', 'blimp__distractor_agreement_relative_clause', 'blimp__drop_argument', 'blimp__ellipsis_n_bar_1', 'blimp__ellipsis_n_bar_2', 'blimp__existential_there_object_raising', 'blimp__existential_there_quantifiers_1', 'blimp__existential_there_quantifiers_2', 'blimp__existential_there_subject_raising', 'blimp__expletive_it_object_raising', 'blimp__inchoative', 'blimp__intransitive', 'blimp__irregular_past_participle_adjectives', 'blimp__irregular_past_participle_verbs', 'blimp__irregular_plural_subject_verb_agreement_1', 'blimp__irregular_plural_subject_verb_agreement_2', 'blimp__left_branch_island_echo_question', 'blimp__left_branch_island_simple_question', 'blimp__matrix_question_npi_licensor_present', 'blimp__npi_present_1', 'blimp__npi_present_2', 'blimp__only_npi_licensor_present', 'blimp__only_npi_scope', 'blimp__passive_1', 'blimp__passive_2', 'blimp__principle_a_c_command', 'blimp__principle_a_case_1', 'blimp__principle_a_case_2', 'blimp__principle_a_domain_1', 'blimp__principle_a_domain_2', 'blimp__principle_a_domain_3', 'blimp__principle_a_reconstruction', 'blimp__regular_plural_subject_verb_agreement_1', 'blimp__regular_plural_subject_verb_agreement_2', 'blimp__sentential_negation_npi_licensor_present', 'blimp__sentential_negation_npi_scope', 'blimp__sentential_subject_island', 'blimp__superlative_quantifiers_1', 'blimp__superlative_quantifiers_2', 'blimp__tough_vs_raising_1', 'blimp__tough_vs_raising_2', 'blimp__transitive', 'blimp__wh_island', 'blimp__wh_questions_object_gap', 'blimp__wh_questions_subject_gap', 'blimp__wh_questions_subject_gap_long_distance', 'blimp__wh_vs_that_no_gap', 'blimp__wh_vs_that_no_gap_long_distance', 'blimp__wh_vs_that_with_gap', 'blimp__wh_vs_that_with_gap_long_distance', 'boolq', 'cb', 'commonsenseqa', 'copa', 
            #'crowspairs', 
            #'cycledletters', 
            'diagnosisbrand', 'diagnosiscity', 'diagnosiscountry', 'diagnosisname', 'diagnosispos1', 'diagnosispos2', 'diagnosispos3', 'diagnosispos4', 
            # 'ethoszeroshot', FileNotFoundError: [Errno 2] No such file or directory: '/private/home/tbmihaylov/fairseq-xlmg/examples/few_shot/data/Ethos/outputs/data/zero_shot_results.csv'
            'hellaswag', 'mnlimatched', 'mnlimismatched', 
            # 'multiplication2digit', 
            'multirc', 'openbookqa', 'pawsx', 'piqa', 
            #'record', 
            #'reversedwords', 
            'rte', 'satanalogies', 
            #'singledigit3ops', 
            'snli', 'storycloze', 
            #'subtraction2digit', 'subtraction3digit', 'subtraction4digit', 'subtraction5digit', 'subtraction6digit', 'sumofdigits', 'symbolinsertion', 
            'wic', 'winograd', 'xwinograd', 'winogrande', 'wsc', 'xcopa', 'xnli']
    })
    
    # model settings
    available_model_settings = default_model_settings.copy()
    model_run_groups = default_model_run_groups.copy()
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for random and majority baselines")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/baselines")

    # set default dir for results
    args = parser.parse_args()

    #args.dry_run = True
    args.local = True
    args.tasks = ["multichoice_tasks"]

    for run_modes, predictor_name in [
        (["random"], "random"),
        (["majority"], "majorityclass"),
    ]:
        args.models = run_modes
        args.predictor_name = predictor_name

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
