from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from inspect import isabstract, signature
import re
from typing import Optional, Dict, Any

import numpy as np

TEMPLATES_REGISTRY = {}
from string import Template
try:
    from jinja2 import Template as JinjaTemplate
except:
    raise ImportError("Could not import jinja2. Use `pip install Jinja2==2.11.3`")

def get_template_class_by_name(template_name):
    return TEMPLATES_REGISTRY[template_name.lower()]


def get_all_templates():
    return list(TEMPLATES_REGISTRY.keys())


@dataclass
class FewShotTemplate(ABC):
    task_description: Optional[str] = None
    lang_code: Optional[str] = "en"

    @abstractmethod
    def encode(self, sample):
        raise NotImplementedError()

    def verbalize(self, sample, candidate):
        return candidate

    def postprocess(self, sample, candidate):
        return candidate

    def encode_correct_candidate(self, sample):
        return self.encode(sample).replace(
            "<mask>", self.verbalize(sample, sample.correct_candidates[0])
        )  # Picking the first correct candidate

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        return cls(
            **{k: v for k, v in kwargs.items() if k in signature(cls).parameters}
        )

    @classmethod
    def get_template_name(cls):
        [name] = re.match("(.+)Template", cls.__name__).groups()
        return name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        template_name = cls.get_template_name()
        assert (
            template_name not in TEMPLATES_REGISTRY
        ), f"{template_name} template already registered!"
        TEMPLATES_REGISTRY[template_name] = cls


@dataclass
class HellaSwagTemplate(FewShotTemplate):
    field_sep: str = " "

    def encode(self, sample):
        activity_label = sample["activity_label"].strip()
        ctx_a = sample["ctx_a"].strip()
        ctx_b = sample["ctx_b"].strip()
        if ctx_b:
            ctx_b = ctx_b[0].upper() + ctx_b[1:]

        field_sep = self.field_sep
        ctx = (ctx_a + field_sep + ctx_b).strip()
        prompt = activity_label + ": " + ctx
        prompt = field_sep.join(prompt.split())
        return prompt + field_sep + "<mask>"


@dataclass
class StoryClozeTemplate(FewShotTemplate):
    field_sep: str = " "
    def encode(self, sample):
        return self.field_sep.join([sample[f"InputSentence{i}"] for i in range(1, 5)] + ["<mask>"])


@dataclass
class WinogradTemplate(FewShotTemplate):
    def encode(self, sample):
        return sample["txt1"] + " <mask> " + sample["txt2"]


@dataclass
class WinograndeTemplate(FewShotTemplate):
    def encode(self, sample):
        return sample["sentence"].replace("_", "<mask>")


def strip_eos(sent):
    sent = sent.rstrip()
    if sent.endswith(".") or sent.endswith("。"):
        sent = sent[:-1]
    return sent


def decapitalize(sent):
    # decapitalize
    sent = sent[0].lower() + (sent[1:] if len(sent) > 1 else "")
    return sent


def capitalize(sent):
    # capitalzie
    sent = sent[0].upper() + (sent[1:] if len(sent) > 1 else "")
    return sent


@dataclass
class XCOPATemplate(FewShotTemplate):
    lang_to_template = {
        "default": {
            # last 2 params are premise and choice pre-processing functions
            "cause": "{premise} {choice}",
            "effect": "{premise} {choice}",
        },
        "__en": {
            "cause": "{premise} because {choice}",
            "effect": "{premise} so {choice}",
        },
        "__it": {
            "cause": "{premise} perché {choice}",
            "effect": "{premise} e quindi {choice}",
        },
        "__vi": {
            "cause": "{premise} bởi vì {choice}",
            "effect": "{premise} thế nên {choice}",
        },
        "__zh": {
            "cause": "{premise}因为{choice}",
            "effect": "{premise}所以{choice}",
        },
        "__tr": {
            "cause": "{premise} çünkü {choice}",
            "effect": "{premise}, o nedenle {choice}",
        },
        "__et": {
            "cause": "{premise}, sest {choice}",
            "effect": "{premise} nii {choice}",
        },
        "__ht": {
            "cause": "{premise} paske {choice}",
            "effect": "{premise} konsa {choice}",
        },
        "__id": {
            "cause": "{premise} karena {choice}",
            "effect": "{premise} begitu {choice}",
        },
        "__sw": {
            "cause": "{premise} kwa sababu {choice}",
            "effect": "{premise} hivyo {choice}",
        },
        "__ta": {
            "cause": "{premise} ஏனெனில் {choice}",
            "effect": "{premise} எனவே {choice}",
        },
        "__th": {
            "cause": "{premise} เพราะ {choice}",
            "effect": "{premise} ดังนั้น {choice}",
        },
    }
    # TODO: "et", "ht", "id", "sw", "ta", "th" (currently using templates translated from English by Google)
    # TODO: "qu" (currently using the default template because Google translate does not cover it)
    # DONE: "it", "vi", "zh", "tr"

    def get_template_processing(self, lang, copa_type):
        template_settings = self.lang_to_template.get(
            "__" + lang, self.lang_to_template.get("default")
        )[copa_type]
        template, premise_proc, choice_proc = (
            template_settings,
            strip_eos,
            decapitalize,
        )

        if isinstance(template_settings, tuple):
            template = template_settings[0]
            if len(template_settings) > 1:
                premise_proc = template_settings[1]
            if len(template_settings) > 2:
                choice_proc = template_settings[2]

        return template, premise_proc, choice_proc

    def encode(self, sample):
        copa_type = sample["question"]
        lang = sample["lang"]
        template, premise_proc, _ = self.get_template_processing(lang, copa_type)

        premise = sample["premise"]
        premise = premise_proc(premise) if premise_proc is not None else premise
        encoded_prompt = template.replace("{premise}", premise).replace(
            "{choice}", "<mask>"
        )
        encoded_prompt = encoded_prompt.strip()  # this strips empty space in the beginning (in case of calibration). 

        return encoded_prompt

    def verbalize(self, sample, candidate):
        lang = sample["lang"]
        copa_type = sample["question"]
        _, _, choice_proc = self.get_template_processing(lang, copa_type)

        choice = sample[candidate]
        choice = choice_proc(choice) if choice_proc is not None else choice

        return choice


class XCOPAMTTemplate(XCOPATemplate):
    lang_to_template = {
        "default": {
            # last 2 params are premise and choice pre-processing functions
            "cause": "{premise} {choice}",
            "effect": "{premise} {choice}",
        },
        "__en": {
            "cause": "{premise} because {choice}",
            "effect": "{premise} so {choice}",
        },
        "__it": {
            "cause": "{premise} perché {choice}",
            "effect": "{premise} quindi {choice}",
        },
        "__vi": {
            "cause": "{premise} bởi vì {choice}",
            "effect": "{premise} nên {choice}",
        },
        "__zh": {
            "cause": "{premise}因为{choice}",
            "effect": "{premise}所以{choice}",
        },
        "__tr": {
            "cause": "{premise} çünkü {choice}",
            "effect": "{premise} yani {choice}",
        },
        "__et": {
            "cause": "{premise}, sest {choice}",
            "effect": "{premise} nii {choice}",
        },
        "__ht": {
            "cause": "{premise} paske {choice}",
            "effect": "{premise} konsa {choice}",
        },
        "__id": {
            "cause": "{premise} karena {choice}",
            "effect": "{premise} begitu {choice}",
        },
        "__sw": {
            "cause": "{premise} kwa sababu {choice}",
            "effect": "{premise} hivyo {choice}",
        },
        "__ta": {
            "cause": "{premise} ஏனெனில் {choice}",
            "effect": "{premise} எனவே {choice}",
        },
        "__th": {
            "cause": "{premise} เพราะ {choice}",
            "effect": "{premise} ดังนั้น {choice}",
        },
    }
    # TODO: "qu" (currently using the default template because Google translate does not cover it)

    @classmethod
    def get_template_name(cls):
        return "xcopa_mt"


class XCOPAEnTemplate(XCOPATemplate):

    def get_template_processing(self, lang, copa_type):
        template_settings = self.lang_to_template.get("__en")[copa_type]
        template, premise_proc, choice_proc = (
            template_settings,
            strip_eos,
            decapitalize,
        )

        if isinstance(template_settings, tuple):
            template = template_settings[0]
            if len(template_settings) > 1:
                premise_proc = template_settings[1]
            if len(template_settings) > 2:
                choice_proc = template_settings[2]

        return template, premise_proc, choice_proc

    @classmethod
    def get_template_name(cls):
        return "xcopa__en"


class XCOPASimpleTemplate(XCOPATemplate):

    def get_template_processing(self, lang, copa_type):
        template_settings = self.lang_to_template.get("default")[copa_type]
        template, premise_proc, choice_proc = (
            template_settings,
            strip_eos,
            decapitalize,
        )

        if isinstance(template_settings, tuple):
            template = template_settings[0]
            if len(template_settings) > 1:
                premise_proc = template_settings[1]
            if len(template_settings) > 2:
                choice_proc = template_settings[2]

        return template, premise_proc, choice_proc

    @classmethod
    def get_template_name(cls):
        return "xcopa_simple"


@dataclass
class StereoSetTemplate(FewShotTemplate):
    field_sep: str = " "

    def encode(self, sample):
        prompt = sample["context"].rstrip()

        return prompt + self.field_sep + "<mask>"

    def verbalize(self, sample, candidate):
        label2sent = {
            sent["gold_label"]: sent["sentence"] for sent in sample["sentences"]
        }
        sent = label2sent[candidate]
        return sent


@dataclass
class CrowSPairsTemplate(FewShotTemplate):
    def encode(self, sample):
        prompt = sample["prompt"].rstrip()
        return prompt + " " + "<mask>"

    def verbalize(self, sample, candidate):
        return sample[candidate]


@dataclass
class EthosZeroShotTemplate(FewShotTemplate):
    def encode(self, sample):
        return sample["prompt"] + " <mask>"

    def verbalize(self, sample, candidate):
        return candidate


@dataclass
class COPATemplate(FewShotTemplate):
    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def encode(self, sample):
        premise = sample["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]

        if sample["question"] == "effect":
            conjunction = self.effect_conj
            # conjunction = self.cause_conj
        elif sample["question"] == "cause":
            conjunction = self.cause_conj
            # conjunction = self.effect_conj
        else:
            raise NotImplementedError

        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()

        return prompt + "<mask>"

    def verbalize(self, sample, candidate):
        def capitalize(c):
            if self.capitalization == "correct":
                words = c.split(" ")
                if words[0] != "I":
                    words[0] = words[0].lower()
                return " ".join(words)
            elif self.capitalization == "bug":
                return c
            elif self.capitalization == "upper":
                return c.upper()
            elif self.capitalization == "lower":
                return c.lower()
            else:
                raise NotImplementedError

        return capitalize(sample[candidate])


@dataclass
class PIQATemplate(FewShotTemplate):
    def encode(self, sample):
        return sample["goal"].strip() + " <mask>"

    def verbalize(self, sample, candidate):
        return sample[candidate]


class ExamsTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["question"]["stem"]
        prompt = self.train_sep.join(
            [
                f"{question}",
                f"<mask>",
            ]
        )
        return prompt

    def verbalize(self, sample, candidate):
        def verbalize_choice(choice):
            choice_text = choice["text"]
            if not sample["question"]["stem"].endswith("?"):
                choice_text = capitalize(choice_text)
            if not choice_text.endswith("."):
                choice_text = choice_text + "."

            para = [choice["para"]] if "para" in choice else []

            return self.train_sep.join([choice_text] + para)

        label2choice = {
            choice["label"]: choice for choice in sample["question"]["choices"]
        }
        choice = label2choice[candidate]
        verbalized = verbalize_choice(choice)
        return verbalized


def class_name_to_readable_template_name(name):
    re_callback = lambda pat: "_" + pat.group(1).lower()
    name = re.sub(r"([A-Z, 0-9]+)", re_callback, name)
    name = name[1:] if name.startswith("_") else name
    return name


@dataclass
class ARCTemplate(FewShotTemplate):
    train_sep: str = " "

    task_descr_question: str = ""  # "Please, answer the following question."
    task_descr_sentence: str = ""  # "Please, continue the following sentence."

    def question_choice_format(self, choice_str):
        # If question ends with ?
        return choice_str.lower()

    def sent_choice_format(self, choice_str):
        # If question is open sentence
        return choice_str.lower()

    def noquestion_choice_format(self, choice_str):
        # If question is empty
        return choice_str.lower()

    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        # Add task description
        task_descr = self.task_descr_question
        if not question.endswith("?"):
            # sentence continuation can have different description
            task_descr = self.task_descr_sentence

        fields = []
        if len(task_descr) > 0:
            fields.append(task_descr)

        if len(question) == 0:
            # If this is a calibration example.
            return "<mask>"
        elif question.endswith("?"):
            fields.extend([question, "<mask>"])
            return self.train_sep.join(fields)
        else:
            fields.extend([question + " <mask>"])
            return self.train_sep.join(fields)

    def verbalize(self, sample, candidate):
        question = sample["question"]["stem"].strip()

        def clear_choice(choice_str):
            if len(question) == 0:
                # Calibration
                choice_str = self.noquestion_choice_format(choice_str)
            elif question.endswith("?"):
                # Question
                choice_str = self.question_choice_format(choice_str)
            else:
                # Sentence continuation
                choice_str = self.sent_choice_format(choice_str)

            if not choice_str.endswith("."):
                choice_str = choice_str + "."

            return choice_str

        label2text = {
            choice["label"]: choice["text"] for choice in sample["question"]["choices"]
        }

        choice = label2text[candidate]
        choice = clear_choice(choice)
        return choice

    @classmethod
    def get_template_name(cls):
        [name] = re.match("(.+)Template", cls.__name__).groups()

        return class_name_to_readable_template_name(name)


@dataclass
class ArcNoChoiceLowercaseTemplate(ARCTemplate):
    def question_choice_format(self, choice_str):
        return choice_str

    def noquestion_choice_format(self, choice_str):
        return choice_str

    def sent_choice_format(self, choice_str):
        return choice_str


@dataclass
class ArcStruct1Template(ARCTemplate):
    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        # Add task description
        task_descr = self.task_descr_question
        if not question.endswith("?"):
            # sentence continuation can have different description
            task_descr = self.task_descr_sentence

        if len(question) == 0:
            # If this is a calibration example.
            return "<mask>"
        elif question.endswith("?"):
            return self.train_sep.join(
                [task_descr, f"Question: {question}", "Answer: <mask>"]
            )
        else:
            return self.train_sep.join([task_descr, question + " <mask>"])


@dataclass
class ArcStruct2Template(ARCTemplate):
    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        # Add task description
        task_descr = self.task_descr_question
        if not question.endswith("?"):
            # sentence continuation can have different description
            task_descr = self.task_descr_sentence

        if len(question) == 0:
            # If this is a calibration example.
            return "<mask>"
        elif question.endswith("?"):
            return self.train_sep.join([task_descr, f"Q: {question}", "A: <mask>"])
        else:
            return self.train_sep.join([task_descr, question + " <mask>"])

@dataclass
class SentimentAnalysisTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        prompt = self.train_sep.join(
            [
                f'{sample["sentence"]} .',
                "All in all, it was <mask>",
            ]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return {"0": "bad", "1": "good"}[
            candidate
        ]

@dataclass
class ArcChallengeGPT3Template(ARCTemplate):
    """
        ARC Challenge Template according to gpt3 paper - http://arxiv.org/abs/2005.14165

    Template verbalization example (lines without #):
        ### Prompt for cand `A`:
        Question: Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as
        Answer: deep sea animals.
        ##########
        ### Calibrations prompts for cand `A`:
        ## calib option 0 prompt:
        Answer: deep sea animals.
    """

    field_sep = " s"

    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        if len(question) == 0:
            # If this is a calibration example.
            return "Answer: <mask>"
        else:
            return self.field_sep.join([f"Question: {question}", "Answer: <mask>"])

    @classmethod
    def get_template_name(cls):
        return "arcchallenge_gpt3"


@dataclass
class OpenbookqaGPT3CalibV1Template(ARCTemplate):
    """
        OpenbookQA Template according to gpt3 paper - http://arxiv.org/abs/2005.14165
        Using "Answer:" as calibration answer_context. The paper states that they use either Answer: or A:

    Template verbalization example (lines without #):
        ### Prompt for cand `A`:
        Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as
        deep sea animals.
        ##########
        ### Calibrations prompts for cand `A`:
        ## calib option 0 prompt:
        Answer: deep sea animals.
    """

    field_sep = " "

    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        if len(question) == 0:
            # If this is a calibration example.
            return "Answer: <mask>"
        else:
            return self.field_sep.join([f"{question}", "<mask>"])

    @classmethod
    def get_template_name(cls):
        return "openbookqa_gpt3_calib_v1"


@dataclass
class OpenbookqaGPT3CalibV2Template(ARCTemplate):
    """
        OpenbookQA Template according to gpt3 paper - http://arxiv.org/abs/2005.14165
        Using "A:" as calibration answer_context. The paper states that they use either Answer: or A:

    Template verbalization example (lines without #):
        ### Prompt for cand `A`:
        Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as deep sea animals.
        ##########
        ### Calibrations prompts for cand `A`:
        ## calib option 0 prompt:
        A: deep sea animals.
    """

    field_sep = " "

    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        if len(question) == 0:
            # If this is a calibration example.
            return "A: <mask>"
        else:
            return self.field_sep.join([f"{question}", "<mask>"])

    @classmethod
    def get_template_name(cls):
        return "openbookqa_gpt3_calib_v2"


@dataclass
class OpenbookqaGPT3CalibEmptyTemplate(ARCTemplate):
    """
        OpenbookQA Template according to gpt3 paper - http://arxiv.org/abs/2005.14165
        Using empty calibration answer_context.

        The paper states that they use either Answer: or A: but we also want to test the empty context.

    Template verbalization example (lines without #):
        ### Prompt for cand `A`:
        Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as deep sea animals.
        ##########
        ### Calibrations prompts for cand `A`:
        ## calib option 0 prompt:
        deep sea animals.
    """

    field_sep = " "

    def encode(self, sample):
        question = sample["question"]["stem"].strip()

        if len(question) == 0:
            # If this is a calibration example.
            return "<mask>"
        else:
            return self.field_sep.join([f"{question}", "<mask>"])

    @classmethod
    def get_template_name(cls):
        return "openbookqa_gpt3_calib_empty"


@dataclass
class ArcCapitalizeChoiceTemplate(ARCTemplate):
    def question_choice_format(self, choice_str):
        return capitalize(choice_str.lower())


@dataclass
class ArcCalibFormat1Template(ARCTemplate):
    def noquestion_choice_format(self, choice_str):
        return "It is " + choice_str.lower()


@dataclass
class ArcCalibFormat2Template(ARCTemplate):
    def noquestion_choice_format(self, choice_str):
        return "This is " + choice_str.lower()


@dataclass
class ArcCalibFormat3Template(ARCTemplate):
    def noquestion_choice_format(self, choice_str):
        return "The answer is " + choice_str.lower()


@dataclass
class ArcDescr1Template(ARCTemplate):
    task_descr_question: str = "Answer the question."
    task_descr_sentence: str = "Continue the sentence."


@dataclass
class ArcDescr2Template(ARCTemplate):
    task_descr_question: str = "Answer the following question."
    task_descr_sentence: str = "Continue the following sentence."


@dataclass
class ArcDescr3Template(ARCTemplate):
    task_descr_question: str = "Please, answer the following question."
    task_descr_sentence: str = "Please, continue the following sentence."


@dataclass
class ArcChoiceFormat1Template(ARCTemplate):
    def question_choice_format(self, choice_str):
        return "It is " + choice_str.lower()


@dataclass
class ArcChoiceFormat2Template(ARCTemplate):
    def question_choice_format(self, choice_str):
        return "This is " + choice_str.lower()


@dataclass
class ArcChoiceFormat3Template(ARCTemplate):
    def question_choice_format(self, choice_str):
        return "The answer is " + choice_str.lower()


@dataclass
class Arc1Template(ARCTemplate):
    def question_choice_format(self, choice_str):
        return "It is " + choice_str.lower()

    def sent_choice_format(self, choice_str):
        return choice_str.lower()

    def noquestion_choice_format(self, choice_str):
        return "It is " + choice_str.lower()


@dataclass
class ARCOldTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["question"]["stem"].strip()
        if len(question) == 0:
            return "<mask>"
        else:
            return question + " <mask>"

    def verbalize(self, sample, candidate):
        question = sample["question"]["stem"].strip()

        def clear_choice(choice_str):
            choice_str = choice_str.lower()
            if question.endswith("?") or len(question) == 0:
                # choice_str = capitalize(choice_str)
                choice_str = "It is " + choice_str

            if not choice_str.endswith("."):
                choice_str = choice_str + "."

            return choice_str

        label2text = {
            choice["label"]: choice["text"] for choice in sample["question"]["choices"]
        }

        choice = label2text[candidate]
        choice = clear_choice(choice)
        return choice

    @classmethod
    def get_template_name(cls):
        return "arc_old"


@dataclass
class OpenBookQATemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["question"]["stem"].strip()
        return question + " <mask>"
    
    def verbalize(self, sample, candidate):
        
        def clear_choice(choice_str):
            choice_str = choice_str.strip()
            if not choice_str.endswith("."):
                choice_str = choice_str + "."
            return choice_str

        label2text = {
            choice["label"]: choice["text"] for choice in sample["question"]["choices"]
        }

        choice = label2text[candidate]
        choice = clear_choice(choice)
        return choice

    @classmethod
    def get_template_name(cls):
        return "openbookqa"


@dataclass
class ODQATemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["question"]
        prompt = self.train_sep.join(
            [
                f"Q: {question}",
                f"A: <mask>",
            ]
        )
        return prompt

@dataclass
class GenerateTextTemplate(FewShotTemplate):
    field_sep: str = " "

    def encode(self, sample):
        text_start = sample["text_prompt"]
        prompt = self.field_sep.join(
            [
                f"{text_start}",
                f"<mask>",
            ]
        )
        return prompt



@dataclass
class WiCTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        word = sample["word"]
        sentence1 = sample["sentence1"]
        sentence2 = sample["sentence2"]
        prompt = self.train_sep.join(
            [
                f"Sentence 1: {sentence1}",
                f"Sentence 2: {sentence2}",
                f"Question: Is the word '{word}' used in the same way in the two sentences above?",
                "Answer: <mask>",
            ]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return {"false": "False", "true": "True"}[candidate]


@dataclass
class BoolQTemplate(FewShotTemplate):
    train_sep: str = " "
    capitalization = "correct"

    def encode(self, sample):
        question = sample["question"]
        passage = sample["passage"]

        if self.capitalization == "correct" and len(question) > 0:
            question = question.capitalize().strip("?") + "?"

        paragraph_strings = (
            [f"Paragraph: {passage}"] if len(passage) > 0 else []
        )  # Paragraph-free when used in empty dummy example
        question_strings = (
            [f"Question: {question}"] if len(question) > 0 else []
        )  # Question-free when used in empty dummy example

        prompt = self.train_sep.join(
            paragraph_strings + question_strings + ["Answer: <mask>"]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return {"false": "False", "true": "True"}[candidate]


def get_field_verbalization_strings(templates_settings, field_key, field_value):
    """Generate field strings given the field value and the field templates.

    Args:
        templates_settings (dict[str, Any]): Templates settings.
        field_key (str): The field to generate strings for.
        field_value (str): The value to replace the template with.

    Returns:
        List[str]: List of strings for the field.
                    These are combined with all fields converted to text field_sep.join().
    """

    if len(field_value) == 0:
        # The value is empty so the field will not be generated.
        # This is used in calibration sample.
        return []

    field_settings = templates_settings[field_key]
    if isinstance(field_settings, tuple) and len(field_settings) > 1:
        field_template, field_preprocess = field_settings
        if field_preprocess is not None:
            field_value = field_preprocess(field_value)
    else:
        field_template = field_settings

    field_value_string = field_template.replace("{" + field_key + "}", field_value)
    return [field_value_string]



@dataclass
class GenerativeNLIMultilingualBaseTemplate(FewShotTemplate):
    """Generative XNLI template where the fields are in English.

    Example prompts:
    ### Prompt for cand `entailment`:
    And he said, Mama, I'm home, right? Yes, he called his mom as soon as the school bus dropped him off.
    """

    def process1(self, s):
        s_original = s
        s = s.strip()
        if len(s) == 0:
            return s

        if s[0].islower():
            s = s[0].upper() + s[1:]
        while len(s) > 0 and s[-1] in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
            s = s[:-1].strip()
        
        if len(s) == 0:
            print(f"Warning: sequence '{s_original}' length equal to 0 after processing")

        return s

    def process2(self, s):
        s = s.strip()
        if len(s) == 0:
            return s
            
        if s[0].isupper() and (s[0] != "I" or (s[1] not in "' ,")):
            s = s[0].lower() + s[1:]
        return s

    field_sep: str = " "

    language_templates_settings = {
        "default": {  # the last param is pre-processing function.
            "sentence1": ("{sentence1}, right?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            # "answer": "True, False, or Neither? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {
                "entailment": "Yes",
                "contradiction": "No",
                "neutral": "Also",
            },
        },
    }

    @abstractproperty
    def fixed_lang_code(self):
        return None

    def get_template_settings(self, lang_code: str):
        """Determine the current template settings.
        Args:
            lang_code ([str]): Lang code to use. This is ignored if self.fixed_lang_code is not None!

        Returns:
            Dict[str, Any]: Template settings
        """
        curr_lang_code = (
            lang_code if self.fixed_lang_code is None else self.fixed_lang_code
        )
        template_settings = self.language_templates_settings.get(
            "__" + curr_lang_code, self.language_templates_settings["default"]
        )

        return template_settings

    def encode(self, sample):
        lang_code = sample["language"]
        templates_settings = self.get_template_settings(lang_code)

        sentence1_strings = get_field_verbalization_strings(
            templates_settings, "sentence1", self.process1(sample["sentence1"])
        )
        sentence2_strings = get_field_verbalization_strings(
            templates_settings, "sentence2", self.process2(sample["sentence2"])
        )
        # answer_strings = get_field_verbalization_strings(
        #     templates_settings, "answer", "<mask>"
        # )

        field_sep = templates_settings.get("field_sep", self.field_sep)
        prompt = field_sep.join(sentence1_strings + sentence2_strings)
        return prompt

    def verbalize(self, sample, candidate):
        lang_code = sample["language"]
        template_settings = self.get_template_settings(lang_code)
        label_map = template_settings["label_map"]
        
        return label_map[candidate]

    @classmethod
    def get_template_name(cls):
        return "xnli_generativenli_base"


@dataclass
class PawsXTemplate(FewShotTemplate):
    """This is the default PawsX template where the fields are in English.

    Example prompts (DE):

        ### Prompt for cand `true`:
        Sentence 1: Something happened. Sentence 2: Nothing happened. Sentence 1 and Sentence 2 have the same meaning. True or False? False
        ##########
        ### Calibrations prompts for cand `true`:
        ## calib option 0 prompt:
        Sentence 1 and Sentence 2 have the same meaning. True or False? False
    """

    field_sep: str = " "

    language_templates_settings = {
        # "de", "en", "es", "fr", "ja", "ko", "zh"
        "default": {
            # the last param is pre-processing function.
            "sentence1": ("Sentence 1: {sentence1}", None),
            "sentence2": ("Sentence 2: {sentence2}", None),
            "answer": "Sentence 1 and Sentence 2 have the same meaning. True or False? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {"true": "True", "false": "False"},
        },
    }

    @abstractproperty
    def fixed_lang_code(self):
        return None

    def get_template_settings(self, lang_code: str):
        """Determine the current template settings.
        Args:
            lang_code ([str]): Lang code to use. This is ignored if self.fixed_lang_code is not None!

        Returns:
            Dict[str, Any]: Template settings
        """
        curr_lang_code = (
            lang_code if self.fixed_lang_code is None else self.fixed_lang_code
        )
        template_settings = self.language_templates_settings.get(
            "__" + curr_lang_code, self.language_templates_settings["default"]
        )

        return template_settings

    def encode(self, sample):
        lang_code = sample["lang"]
        templates_settings = self.get_template_settings(lang_code)

        sentence1_strings = get_field_verbalization_strings(
            templates_settings, "sentence1", sample["sentence1"]
        )
        sentence2_strings = get_field_verbalization_strings(
            templates_settings, "sentence2", sample["sentence2"]
        )
        answer_strings = get_field_verbalization_strings(
            templates_settings, "answer", "<mask>"
        )

        field_sep = templates_settings.get("field_sep", self.field_sep)
        prompt = field_sep.join(sentence1_strings + sentence2_strings + answer_strings)
        return prompt

    def verbalize(self, sample, candidate):
        lang_code = sample["lang"]
        template_settings = self.get_template_settings(lang_code)
        label_map = template_settings["label_map"]

        return label_map[candidate]


@dataclass
class PawsXENTemplate(PawsXTemplate):
    @property
    def fixed_lang_code(self):
        return "en"

    @classmethod
    def get_template_name(cls):
        return "pawsx__en"


@dataclass
class PawsXMTTemplate(PawsXTemplate):
    """
        This template is defined using Google Translate to translate the fields from the default English template.

    Example prompts (DE):

        ### Prompt for cand `true`:
        Satz 1: Durch die Zusammenlegung des Four Rivers Council und des Audubon Council entstand der Shawnee Trails Council.
        Satz 2: Shawnee Trails Council entstand durch die Fusion zwischen dem Four Rivers Council und dem Audubon Council.
        Satz 1 und Satz 2 haben die gleiche Bedeutung. Richtig oder falsch? Richtig
        ##########
        ### Calibrations prompts for cand `true`:
        ## calib option 0 prompt:
        Satz 1 und Satz 2 haben die gleiche Bedeutung. Richtig oder falsch? Richtig
    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # "de", "en", "es", "fr", "ja", "ko", "zh"
        "default": {
            # the last param is pre-processing function.
            "sentence1": ("Sentence 1: {sentence1}", None),
            "sentence2": ("Sentence 2: {sentence2}", None),
            "answer": "Sentence 1 and Sentence 2 have the same meaning. True or False? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {"true": "True", "false": "False"},
        },
        "__de": {
            "sentence1": ("Satz 1: {sentence1}", None),
            "sentence2": ("Satz 2: {sentence2}", None),
            "answer": "Satz 1 und Satz 2 haben die gleiche Bedeutung. Richtig oder falsch? <mask>",
            "label_map": {"true": "Richtig", "false": "Falsch"},
        },  # translated with Google Translate
        "__es": {
            "sentence1": ("Oración 1: {sentence1}", None),
            "sentence2": ("Oración 2: {sentence2}", None),
            "answer": "La oración 1 y la oración 2 tienen el mismo significado. ¿Verdadero o falso? <mask>",
            "label_map": {"true": "Verdadero", "false": "Falso"},
        },  # translated with Google Translate
        "__fr": {
            "sentence1": ("Phrase 1: {sentence1}", None),
            "sentence2": ("Phrase 2: {sentence2}", None),
            "answer": "La phrase 1 et la phrase 2 ont le même sens. Vrai ou faux? <mask>",
            "label_map": {"true": "Vrai", "false": "Faux"},
        },  # translated with Google Translate
        "__ja": {
            "sentence1": ("文1：{sentence1}", None),
            "sentence2": ("文2：{sentence2}", None),
            "answer": "文1と文2は同じ意味です。正しいか間違っているか？ <mask>",
            "label_map": {"true": "本当", "false": "誤り"},
        },  # translated with Google Translate
        "__ko": {
            "sentence1": ("문장 1: {sentence1}", None),
            "sentence2": ("문장 2: {sentence2}", None),
            "answer": "문장 1과 문장 2는 같은 의미입니다. 참인가 거짓인가? <mask>",
            "label_map": {"true": "진실", "false": "그릇된"},
        },  # translated with Google Translate
        "__zh": {
            "sentence1": ("第 1 句：{sentence1}", None),
            "sentence2": ("句子 2：{sentence2}", None),
            "answer": "第 1 句和第 2 句具有相同的含义。对或错？ <mask>",
            "label_map": {"true": "真的", "false": "错误的"},
        },  # translated with Google Translate
    }

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt"


@dataclass
class PawsXConditionalMTTemplate(PawsXTemplate):
    
    @property
    def fixed_lang_code(self):
        return None

    def encode(self, sample):
        lang_code = sample["lang"]
        templates_settings = self.get_template_settings(lang_code)

        sentence1_strings = get_field_verbalization_strings(
            templates_settings, "sentence1", sample["sentence1"]
        )
        sentence2_strings = get_field_verbalization_strings(
            templates_settings, "sentence2", sample["sentence2"]
        )
        answer_strings = get_field_verbalization_strings(
            templates_settings, "answer", "<mask>"
        )

        field_sep = templates_settings.get("field_sep", self.field_sep)
        prompt = field_sep.join(answer_strings + sentence1_strings + sentence2_strings)
        return prompt

    @classmethod
    def get_template_name(cls):
        return "pawsx_conditional_mt"


@dataclass
class PawsXGenerativeNLIMTTemplate(PawsXMTTemplate):
    """
    This template is defined using Google Translate to translate the fields from an English template.
    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # "de", "en", "es", "fr", "ja", "ko", "zh"
        
        "default": {  # the last param is pre-processing function.
            "sentence1": ("{sentence1}, right?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "true": "Yes",
                "false": "No",
                "neutral": "Also",
            },
        },
        '__fr': {'sentence1': ('{sentence1}, droit?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'true': 'Oui', 'false': 'Non', 'neutral': 'Aussi'}},  # copied from XNLI mt
        '__es': {'sentence1': ('{sentence1}, ¿Derecha?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'true': 'sí', 'false': 'No', 'neutral': 'También'}},  # copied from XNLI mt
        '__de': {'sentence1': ('{sentence1}, rechts?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'true': 'Jawohl', 'false': 'Nein', 'neutral': 'Ebenfalls'}},  # copied from XNLI mt
        '__ru': {'sentence1': ('{sentence1}, Правильно?', None), 'label_map': {'true': 'да', 'false': 'Нет', 'neutral': 'Также'}},  # copied from XNLI mt
        '__zh': {'sentence1': ('{sentence1}， 对？', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'true': '是的', 'false': '不', 'neutral': '还'}},  # copied from XNLI mt
        "__ja": {
            "sentence1": ("、 右", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"true": "はい", "false": "の", "neutral": "また"},
        },  # translated with Google Translate
        "__ko": {
            "sentence1": (", 오른쪽?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"true": "예", "false": "아니요", "neutral": "또한"},
        },  # translated with Google Translate
    }

    def encode(self, sample):
        lang_code = sample["lang"]
        templates_settings = self.get_template_settings(lang_code)

        sentence1_strings = get_field_verbalization_strings(
            templates_settings, "sentence1", sample["sentence1"]
        )
        sentence2_strings = get_field_verbalization_strings(
            templates_settings, "sentence2", sample["sentence2"]
        )
        answer_strings = []

        field_sep = templates_settings.get("field_sep", self.field_sep)
        prompt = field_sep.join(sentence1_strings + sentence2_strings + answer_strings)
        return prompt

    @classmethod
    def get_template_name(cls):
        return "pawsx_generativenli_mt"


@dataclass
class PawsXGenerativeNLIENTemplate(PawsXGenerativeNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "en"

    @classmethod
    def get_template_name(cls):
        return "pawsx_generativenli__en"


@dataclass
class PawsXGenerativeNLIHTTemplate(PawsXGenerativeNLIMTTemplate):
    """
    This template is create using human translation/edit of the english version.
    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # "de", "en", "es", "fr", "ja", "ko", "zh"
        
        "default": {  # the last param is pre-processing function.
            "sentence1": ("{sentence1}, right?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "true": "Yes",
                "false": "No",
                "neutral": "Also",
            },
        },
        "__fr": {
            "sentence1": ("{sentence1}, je me trompe?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "true": "Oui",
                "false": "Non",
                "neutral": "Également",
            },
        },  # translated with Google Translate then FIXED by human
        "__es": {
            "sentence1": ("{sentence1}, ¿verdad?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "true": "Sí",
                "false": "No",
                "neutral": "Además",
            },
        },  # translated with Google Translate then FIXED by human
        "__de": {
            "sentence1": ("{sentence1}, oder?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "true": "Ja",
                "false": "Nein",
                "neutral": "Außerdem",
            },
        },  # translated with Google Translate then FIXED by human
        "__ru": {
            "sentence1": ("{sentence1}, верно?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "true": "Да",
                "false": "Нет",
                "neutral": "Также",
            },
        },  # translated with Google Translate
        "__zh": {
            "sentence1": ("{sentence1}， 对吗？", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"true": "是的", "false": "不", "neutral": "而且"},
        },  # translated with Google Translate then FIXED by human
        "__ja": {
            "sentence1": ("、 右", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"true": "はい", "false": "の", "neutral": "また"},
        },  # translated with Google Translate
        "__ko": {
            "sentence1": (", 오른쪽?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"true": "예", "false": "아니요", "neutral": "또한"},
        },  # translated with Google Translate
    }

    @classmethod
    def get_template_name(cls):
        return "pawsx_generativenli_ht"


@dataclass
class PawsXMTDETemplate(PawsXMTTemplate):
    @property
    def fixed_lang_code(self):
        return "de"

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt__de"


@dataclass
class PawsXMTESTemplate(PawsXMTTemplate):
    @property
    def fixed_lang_code(self):
        return "es"

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt__es"


@dataclass
class PawsXMTFRTemplate(PawsXMTTemplate):
    @property
    def fixed_lang_code(self):
        return "fr"

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt__fr"


@dataclass
class PawsXMTJATemplate(PawsXMTTemplate):
    @property
    def fixed_lang_code(self):
        return "ja"

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt__ja"


@dataclass
class PawsXMTKOTemplate(PawsXMTTemplate):
    @property
    def fixed_lang_code(self):
        return "ko"

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt__ko"


@dataclass
class PawsXMTZHTemplate(PawsXMTTemplate):
    @property
    def fixed_lang_code(self):
        return "zh"

    @classmethod
    def get_template_name(cls):
        return "pawsx_mt__zh"


@dataclass
class GPT3StyleNLIMultilingualBaseTemplate(FewShotTemplate):
    """This is the default XNLI template where the fields are in English."""

    field_sep: str = " "

    language_templates_settings = {
        "default": {  # the last param is pre-processing function.
            "sentence1": ("Paragraph: {sentence1}", None),
            "sentence2": ("Question: {sentence2}", None),
            "answer": "True, False, or Neither? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {
                "entailment": "True",
                "contradiction": "False",
                "neutral": "Neither",
            },
        },
    }

    @abstractproperty
    def fixed_lang_code(self):
        return None

    def get_template_settings(self, lang_code: str):
        """Determine the current template settings.
        Args:
            lang_code ([str]): Lang code to use. This is ignored if self.fixed_lang_code is not None!

        Returns:
            Dict[str, Any]: Template settings
        """
        curr_lang_code = (
            lang_code if self.fixed_lang_code is None else self.fixed_lang_code
        )
        template_settings = self.language_templates_settings.get(
            "__" + curr_lang_code, self.language_templates_settings["default"]
        )

        return template_settings

    def encode(self, sample):
        lang_code = sample["language"]
        templates_settings = self.get_template_settings(lang_code)

        sentence1_strings = get_field_verbalization_strings(
            templates_settings, "sentence1", sample["sentence1"]
        )
        sentence2_strings = get_field_verbalization_strings(
            templates_settings, "sentence2", sample["sentence2"]
        )
        answer_strings = get_field_verbalization_strings(
            templates_settings, "answer", "<mask>"
        )

        field_sep = templates_settings.get("field_sep", self.field_sep)
        prompt = field_sep.join(sentence1_strings + sentence2_strings + answer_strings)
        return prompt

    def verbalize(self, sample, candidate):
        lang_code = sample["language"]
        template_settings = self.get_template_settings(lang_code)
        label_map = template_settings["label_map"]

        return label_map[candidate]


@dataclass
class GenerativeNLITemplate(FewShotTemplate):
    def encode(self, sample):
        def process1(s):
            s = s.strip()
            if len(s) == 0:
                return s

            if s[0].islower():
                s = s[0].upper() + s[1:]
            while s[-1] in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                s = s[:-1].strip()
            return s

        def process2(s):
            s = s.strip()
            if len(s) == 0:
                return s

            if s[0].isupper() and (s[0] != "I" or (s[1] not in "' ,")):
                s = s[0].lower() + s[1:]
            return s

        premise, hypothesis = sample["premise"], sample["hypothesis"]
        return process1(premise) + ", right? <mask>, " + process2(hypothesis)

    def verbalize(self, sample, candidate):
        return {
            "entailment": "Yes",
            "contradiction": "No",
            "neutral": "Also",
            "not_entailment": "No",
        }[candidate]


@dataclass
class XNLIGenerativeNLIENTemplate(GenerativeNLIMultilingualBaseTemplate):
    @property
    def fixed_lang_code(self):
        return "en"

    @classmethod
    def get_template_name(cls):
        return "xnli_generativenli__en"


@dataclass
class XNLIGenerativeNLIMTTemplate(GenerativeNLIMultilingualBaseTemplate):
    """
    This template is defined by using Google Translate to translate the fields from the default English template.

    Intend to serve as baseline. Please do not modify these templates manually.
    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # ["fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]
        'default': {'sentence1': ('{sentence1}, right?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Yes', 'contradiction': 'No', 'neutral': 'Also'}},
        '__fr': {'sentence1': ('{sentence1}, droit?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Oui', 'contradiction': 'Non', 'neutral': 'Aussi'}},  
        '__es': {'sentence1': ('{sentence1}, ¿Derecha?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'sí', 'contradiction': 'No', 'neutral': 'También'}},  
        '__de': {'sentence1': ('{sentence1}, rechts?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Jawohl', 'contradiction': 'Nein', 'neutral': 'Ebenfalls'}},  
        '__el': {'sentence1': ('{sentence1}, σωστά?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Ναί', 'contradiction': 'Οχι', 'neutral': 'Επίσης'}},  
        '__bg': {'sentence1': ('{sentence1}, нали?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Да', 'contradiction': 'Не', 'neutral': 'Също'}},  
        '__ru': {'sentence1': ('{sentence1}, Правильно?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'да', 'contradiction': 'Нет', 'neutral': 'Также'}},  
        '__tr': {'sentence1': ('{sentence1}, sağ?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Evet', 'contradiction': 'Numara', 'neutral': 'Ayrıca'}},  
        '__ar': {'sentence1': ('{sentence1}، حق؟', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'نعم', 'contradiction': 'لا', 'neutral': 'أيضا'}},  
        '__vi': {'sentence1': ('{sentence1}, đúng?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'đúng', 'contradiction': 'Không', 'neutral': 'Cũng'}},  
        '__th': {'sentence1': ('{sentence1}, ขวา?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'ใช่', 'contradiction': 'เลขที่', 'neutral': 'อีกด้วย'}},  
        '__zh': {'sentence1': ('{sentence1}， 对？', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': '是的', 'contradiction': '不', 'neutral': '还'}},  
        '__hi': {'sentence1': ('{sentence1}, अधिकार?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'हां', 'contradiction': 'नहीं', 'neutral': 'भी'}},  
        '__sw': {'sentence1': ('{sentence1}, haki?', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'Ndio', 'contradiction': 'Hapana', 'neutral': 'Pia'}},  
        '__ur': {'sentence1': ('{sentence1}، ٹھیک ہے؟', None), "sentence2": ("<mask>, {sentence2}", None), 'label_map': {'entailment': 'جی ہاں', 'contradiction': 'نہیں', 'neutral': 'بھی'}},  
    }

    @classmethod
    def get_template_name(cls):
        return "xnli_generativenli_mt"


class XNLIGenerativeNLIHTTemplate(GenerativeNLIMultilingualBaseTemplate):
    """
    This template is defined by first using Google Translate to translate the fields from the default English template,
    then manually verified and corrected by humans.
    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # ["fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]
        "default": {  # the last param is pre-processing function.
            "sentence1": ("{sentence1}, right?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Yes",
                "contradiction": "No",
                "neutral": "Also",
            },
        },
        "__fr": {
            "sentence1": ("{sentence1}, je me trompe?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Oui",
                "contradiction": "Non",
                "neutral": "Également",
            },
        },  
        "__es": {
            "sentence1": ("{sentence1}, ¿verdad?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Sí",
                "contradiction": "No",
                "neutral": "Además",
            },
        },  
        "__de": {
            "sentence1": ("{sentence1}, oder?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Ja",
                "contradiction": "Nein",
                "neutral": "Außerdem",
            },
        }, 
        "__el": {
            "sentence1": ("{sentence1}, αλήθεια?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Ναί",
                "contradiction": "Οχι",
                "neutral": "Επίσης",
            },
        },  
        "__bg": {
            "sentence1": ("{sentence1}, нали?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"entailment": "Да", "contradiction": "Не", "neutral": "Също"},
        }, 
        "__ru": {
            "sentence1": ("{sentence1}, верно?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Да",
                "contradiction": "Нет",
                "neutral": "Также",
            },
        }, 
        "__tr": {
            "sentence1": ("{sentence1}, Sağ?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Evet",
                "contradiction": "Hayır",
                "neutral": "Ayrıca",
            },
        },  
        "__ar": {
            "sentence1": ("{sentence1}، صحيح؟", None),
            "sentence2": ("<mask>، {sentence2}", None),
            "label_map": {
                "entailment": "نعم",
                "contradiction": "لا",
                "neutral": "أيضا",
            },
        },  
        "__vi": {
            "sentence1": ("{sentence1}, đúng không?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Đúng",
                "contradiction": "Không",
                "neutral": "Ngoài ra",
            },
        },  
        "__th": {
            "sentence1": ("{sentence1}, ใช่มั้ย?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "ใช่",
                "contradiction": "ไม่",
                "neutral": "อีกด้วย",
            },
        },  
        "__zh": {
            "sentence1": ("{sentence1}， 对吗？", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"entailment": "是的", "contradiction": "不", "neutral": "而且"},
        },  
        "__hi": {
            "sentence1": ("{sentence1}, सही?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "हाँ",
                "contradiction": "नहीं",
                "neutral": "तथा",
            },
        }, 
        "__sw": {
            "sentence1": ("{sentence1}, haki?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Ndio",
                "contradiction": "Hapana",
                "neutral": "Pia",
            },
        },  
        "__ur": {
            "sentence1": ("{sentence1}، ٹھیک ہے؟", None),
            "sentence2": ("<mask>، {sentence2}", None),
            "label_map": {
                "entailment": "جی ہاں",
                "contradiction": "نہیں",
                "neutral": "بھی",
            },
        }, 
    }

    @classmethod
    def get_template_name(cls):
        return "xnli_generativenli_ht"


@dataclass
class XNLIGenerativeNLISentenceENTemplate(GenerativeNLIMultilingualBaseTemplate):
    """Generative template with separate sentence "Right?"

    Example prompts:
    ### Prompt for cand `entailment`:
    And he said, Mama, I'm home.. Right? Yes, he called his mom as soon as the school bus dropped him off.
    """

    def process1(self, s):
        return s

    def process2(self, s):
        s = s.strip()
        if s[0].isupper() and (s[0] != "I" or (s[1] not in "' ,")):
            s = s[0].lower() + s[1:]
        return s

    field_sep: str = " "

    language_templates_settings = {
        "default": {  # the last param is pre-processing function.
            "sentence1": ("{sentence1}. Right?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            # "answer": "True, False, or Neither? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {
                "entailment": "Yes",
                "contradiction": "No",
                "neutral": "Also",
            },
        },
    }

    @property
    def fixed_lang_code(self):
        return "en"

    @classmethod
    def get_template_name(cls):
        return "xnli_generativenli_sentence__en"


@dataclass
class XNLIGenerativeSentenceNLIMTTemplate(XNLIGenerativeNLISentenceENTemplate):
    """
    This template is defined using Google Translate to translate the fields from the default English template.

    Example prompts:
    And he said, Mama, I'm home. Right? Yes, he called his mom as soon as the school bus dropped him off.
    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # ["fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]
        "default": {  # the last param is pre-processing function.
            "sentence1": ("{sentence1} Right?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            # "answer": "True, False, or Neither? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {
                "entailment": "Yes",
                "contradiction": "No",
                "neutral": "Also",
            },
        },
        "__fr": {
            "sentence1": ("{sentence1} Droite?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Oui",
                "contradiction": "Non",
                "neutral": "Également",
            },
        },  # translated with Google Translate
        "__es": {
            "sentence1": ("{sentence1} ¿Correcto?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Sí",
                "contradiction": "No",
                "neutral": "También",
            },
        },  # translated with Google Translate
        "__de": {
            "sentence1": ("{sentence1} Richtig?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Jawohl",
                "contradiction": "Nein",
                "neutral": "Ebenfalls",
            },
        },  # translated with Google Translate
        "__el": {
            "sentence1": ("{sentence1} Σωστά?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Ναί",
                "contradiction": "Οχι",
                "neutral": "Επίσης",
            },
        },  # translated with Google Translate
        "__bg": {
            "sentence1": ("{sentence1} Нали?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"entailment": "Да", "contradiction": "Не", "neutral": "Също"},
        },  # translated with Google Translate
        "__ru": {
            "sentence1": ("{sentence1} Правильно?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "да",
                "contradiction": "Нет",
                "neutral": "Также",
            },
        },  # translated with Google Translate
        "__tr": {
            "sentence1": ("{sentence1} Doğru?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Evet",
                "contradiction": "Numara",
                "neutral": "Ayrıca",
            },
        },  # translated with Google Translate
        "__ar": {
            "sentence1": ("يمين؟ {sentence1}", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "نعم",
                "contradiction": "رقم",
                "neutral": "أيضا",
            },
        },  # translated with Google Translate
        "__vi": {
            "sentence1": ("{sentence1} Bên phải?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "đúng",
                "contradiction": "Không",
                "neutral": "Cũng thế",
            },
        },  # translated with Google Translate
        "__th": {
            "sentence1": ("{sentence1} ถูกต้อง?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "ใช่",
                "contradiction": "ไม่",
                "neutral": "อีกด้วย",
            },
        },  # translated with Google Translate
        "__zh": {
            "sentence1": ("{sentence1} 对？", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {"entailment": "是的", "contradiction": "不", "neutral": "还"},
        },  # translated with Google Translate
        "__hi": {
            "sentence1": ("{sentence1} सही?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "हाँ",
                "contradiction": "नहीं",
                "neutral": "भी",
            },
        },  # translated with Google Translate
        "__sw": {
            "sentence1": ("{sentence1} Haki?", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "Ndio",
                "contradiction": "Hapana",
                "neutral": "Pia",
            },
        },  # translated with Google Translate
        "__ur": {
            "sentence1": ("ٹھیک ہے؟ {sentence1}", None),
            "sentence2": ("<mask>, {sentence2}", None),
            "label_map": {
                "entailment": "جی ہاں",
                "contradiction": "نہیں",
                "neutral": "بھی",
            },
        },  # translated with Google Translate
    }

    @classmethod
    def get_template_name(cls):
        return "xnli_generativenli_sentence_mt"


@dataclass
class XNLIENTemplate(GPT3StyleNLIMultilingualBaseTemplate):
    @property
    def fixed_lang_code(self):
        return "en"

    @classmethod
    def get_template_name(cls):
        return "xnli__en"


@dataclass
class XNLIMTTemplate(GPT3StyleNLIMultilingualBaseTemplate):
    """
        This template is defined using Google Translate to translate the fields from the default English template.

    Example prompts (Template BG, Train De, Eval En, 1-shot) :
        ### Prompt for cand `entailment`:
        Параграф: The folks at L'academie Internationale des Arts et des Sciences Numeriques have innovated a clever variant on this trick. Въпрос: The people at the school just followed their lead. Вярно, грешно или нито едното? Грешно
        Параграф: And he said, Mama, I'm home. Въпрос: He called his mom as soon as the school bus dropped him off. Вярно, грешно или нито едното? Вярно

    """

    field_sep: str = " "

    @property
    def fixed_lang_code(self):
        return None

    language_templates_settings = {
        # "de", "en", "es", "fr", "ja", "ko", "zh"
        "default": {  # the last param is pre-processing function.
            "sentence1": ("Paragraph: {sentence1}", None),
            "sentence2": ("Question: {sentence2}", None),
            "answer": "True, False, or Neither? <mask>",  # this needs to have <mask> which will be replaced with the label
            "label_map": {
                "entailment": "True",
                "contradiction": "False",
                "neutral": "Neither",
            },
        },
        "__fr": {
            "sentence1": ("Paragraphe: {sentence1}", None),
            "sentence2": ("Question: {sentence2}", None),
            "answer": "Vrai, faux ou ni l'un ni l'autre\xa0? <mask>",
            "label_map": {
                "entailment": "Vrai",
                "contradiction": "Faux",
                "neutral": "Ni",
            },
        },  # translated with Google Translate
        "__es": {
            "sentence1": ("Párrafo: {sentence1}", None),
            "sentence2": ("Pregunta: {sentence2}", None),
            "answer": "¿Verdadero, falso o ninguno? <mask>",
            "label_map": {
                "entailment": "Cierto",
                "contradiction": "Falso",
                "neutral": "Ninguno de los dos",
            },
        },  # translated with Google Translate
        "__de": {
            "sentence1": ("Absatz: {sentence1}", None),
            "sentence2": ("Frage: {sentence2}", None),
            "answer": "Richtig, falsch oder weder? <mask>",
            "label_map": {
                "entailment": "Richtig",
                "contradiction": "Falsch",
                "neutral": "Weder",
            },
        },  # translated with Google Translate
        "__el": {
            "sentence1": ("Παράγραφος: {sentence1}", None),
            "sentence2": ("Ερώτηση: {sentence2}", None),
            "answer": "Αλήθεια, Λάθος ή Ούτε; <mask>",
            "label_map": {
                "entailment": "Αληθής",
                "contradiction": "Ψευδής",
                "neutral": "κανενα απο τα δυο",
            },
        },  # translated with Google Translate
        "__bg": {
            "sentence1": ("Параграф: {sentence1}", None),
            "sentence2": ("Въпрос: {sentence2}", None),
            "answer": "Вярно, грешно или нито едното? <mask>",
            "label_map": {
                "entailment": "Вярно",
                "contradiction": "Грешно",
                "neutral": "Нито едното",
            },
        },  # translated with Google Translate
        "__ru": {
            "sentence1": ("Параграф: {sentence1}", None),
            "sentence2": ("Вопрос: {sentence2}", None),
            "answer": "Верно, неверно или ни один? <mask>",
            "label_map": {
                "entailment": "Истинный",
                "contradiction": "Ложь",
                "neutral": "Ни один",
            },
        },  # translated with Google Translate
        "__tr": {
            "sentence1": ("Paragraf: {sentence1}", None),
            "sentence2": ("Soru: {sentence2}", None),
            "answer": "Doğru, Yanlış veya Hiçbiri? <mask>",
            "label_map": {
                "entailment": "NS",
                "contradiction": "Yanlış",
                "neutral": "Hiç biri",
            },
        },  # translated with Google Translate
        "__ar": {
            "sentence1": ("فقرة: {sentence1}", None),
            "sentence2": ("سؤال: {sentence2}", None),
            "answer": "<mask> صح أم خطأ أم لا؟",
            "label_map": {
                "entailment": "حقيقي",
                "contradiction": "خاطئة",
                "neutral": "لا هذا ولا ذاك",
            },
        },  # translated with Google Translate
        "__vi": {
            "sentence1": ("Đoạn văn: {sentence1}", None),
            "sentence2": ("Câu hỏi: {sentence2}", None),
            "answer": "Đúng, Sai hay Không? <mask>",
            "label_map": {
                "entailment": "Đúng vậy",
                "contradiction": "Sai",
                "neutral": "Không",
            },
        },  # translated with Google Translate
        "__th": {
            "sentence1": ("ย่อหน้า: {sentence1}", None),
            "sentence2": ("คำถาม: {sentence2}", None),
            "answer": "จริง เท็จ หรือเปล่า? <mask>",
            "label_map": {
                "entailment": "จริง",
                "contradiction": "เท็จ",
                "neutral": "ไม่ใช่ทั้งสองอย่าง",
            },
        },  # translated with Google Translate
        "__zh": {
            "sentence1": ("段落：{sentence1}", None),
            "sentence2": ("问题：{sentence2}", None),
            "answer": "对，错，还是都不是？ <mask>",
            "label_map": {
                "entailment": "真的",
                "contradiction": "错误的",
                "neutral": "两者都不",
            },
        },  # translated with Google Translate
        "__hi": {
            "sentence1": ("अनुच्छेद: {sentence1}", None),
            "sentence2": ("सवाल: {sentence2}", None),
            "answer": "सच, झूठा, या नहीं? <mask>",
            "label_map": {
                "entailment": "सत्य",
                "contradiction": "असत्य",
                "neutral": "न",
            },
        },  # translated with Google Translate
        "__sw": {
            "sentence1": ("Aya: {sentence1}", None),
            "sentence2": ("Swali: {sentence2}", None),
            "answer": "Ukweli, Uwongo, au Wala? <mask>",
            "label_map": {
                "entailment": "Kweli",
                "contradiction": "Uongo",
                "neutral": "Wala",
            },
        },  # translated with Google Translate
        "__ur": {
            "sentence1": ("پیراگراف: {sentence1}", None),
            "sentence2": ("سوال: {sentence2}", None),
            "answer": "سچ ، جھوٹ ، یا نہ؟ <ماسک>",
            "label_map": {
                "entailment": "سچ ہے۔",
                "contradiction": "جھوٹا۔",
                "neutral": "نہ ہی",
            },
        },  # translated with Google Translate
    }

    @classmethod
    def get_template_name(cls):
        return "xnli_mt"


@dataclass
class XNLIFRMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "fr"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__fr"


@dataclass
class XNLIESMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "es"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__es"


@dataclass
class XNLIDEMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "de"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__de"


@dataclass
class XNLIELMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "el"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__el"


@dataclass
class XNLIBGMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "bg"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__bg"


@dataclass
class XNLIRUMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "ru"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__ru"


@dataclass
class XNLITRMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "tr"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__tr"


@dataclass
class XNLIARMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "ar"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__ar"


@dataclass
class XNLIVIMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "vi"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__vi"


@dataclass
class XNLITHMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "th"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__th"


@dataclass
class XNLIZHMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "zh"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__zh"


@dataclass
class XNLIHIMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "hi"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__hi"


@dataclass
class XNLISWMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "sw"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__sw"


@dataclass
class XNLIURMTTemplate(XNLIMTTemplate):
    @property
    def fixed_lang_code(self):
        return "ur"

    @classmethod
    def get_template_name(cls):
        return "xnli_mt__ur"


@dataclass
class BoolQNoStructTemplate(BoolQTemplate):
    def encode(self, sample):
        question = sample["question"]
        passage = sample["passage"]

        if self.capitalization == "correct" and len(question) > 0:
            question = question.capitalize().strip("?") + "?"

        paragraph_strings = (
            [f"{passage}"] if len(passage) > 0 else []
        )  # Paragraph-free when used in empty dummy example
        question_strings = (
            [f"{question}"] if len(question) > 0 else []
        )  # Question-free when used in empty dummy example

        prompt = self.train_sep.join(paragraph_strings + question_strings + ["<mask>"])
        return prompt

    @classmethod
    def get_template_name(cls):
        [name] = re.match("(.+)Template", cls.__name__).groups()
        return class_name_to_readable_template_name(name)


@dataclass
class BoolQNoPassageTemplate(BoolQTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["question"]
        if self.capitalization == "correct":
            question = question.capitalize().strip("?") + "?"
        return f"Question: {question}{self.train_sep}Answer: <mask>"


@dataclass
class CBTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["hypothesis"]
        if len(question) > 0:
            question = sample["hypothesis"].capitalize().strip("?") + "?"

        para_strings = (
            [f"Paragraph: {sample['premise']}"] if len(sample["premise"]) > 0 else []
        )
        question_strings = [
            f"Question:{(' ' if len(question) > 0 else '') + question} True, False, or Neither?"
        ]
        prompt = self.train_sep.join(
            para_strings + question_strings + ["Answer: <mask>"]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return {"entailment": "True", "contradiction": "False", "neutral": "Neither"}[
            candidate
        ]

    @classmethod
    def get_template_name(cls):
        [name] = re.match("(.+)Template", cls.__name__).groups()
        return class_name_to_readable_template_name(name)


@dataclass
class CBGPT3ReproduceTemplate(FewShotTemplate):
    field_sep: str = "\n"
    train_sep: str = "\n\n"

    def encode(self, sample):
        question = sample["hypothesis"]
        if len(question) > 0:
            question = sample["hypothesis"].capitalize().strip(".") + "."

        para_strings = [f"{sample['premise']}"] if len(sample["premise"]) > 0 else []
        question_strings = [
            f"question: {(' ' if len(question) > 0 else '') + question} true, false, or neither?"
        ]
        prompt = self.field_sep.join(
            para_strings + question_strings + ["answer: <mask>"]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return {"entailment": "true", "contradiction": "false", "neutral": "neither"}[
            candidate
        ]

    @classmethod
    def get_template_name(cls):
        return "cb_gpt3_reproduce"


@dataclass
class CBNoStructTemplate(CBTemplate):
    def encode(self, sample):
        question = sample["hypothesis"]
        if len(question) > 0:
            question = sample["hypothesis"].capitalize().strip("?") + "?"

        para_strings = [f"{sample['premise']}"] if len(sample["premise"]) > 0 else []
        question_strings = [
            f"{(' ' if len(question) > 0 else '') + question} True, False, or Neither?"
        ]
        prompt = self.train_sep.join(para_strings + question_strings + ["<mask>"])

        return prompt


@dataclass
class RTETemplate(FewShotTemplate):
    train_sep: str = " "
    field_sep: Optional[str] = None

    def encode(self, sample):
        premise_strings = [sample["premise"]] if len(sample["premise"]) > 0 else []
        question_strings = (
            [f"question: {sample['hypothesis']} True or False?"]
            if len(sample["hypothesis"]) > 0
            else []
        )
        field_sep = self.train_sep if self.field_sep is None else self.field_sep
        return field_sep.join(premise_strings + question_strings + ["answer: <mask>"])

    def verbalize(self, sample, candidate):
        return {"entailment": "True", "not_entailment": "False", "": ""}[candidate]


@dataclass
class RTEGPT3ReproduceTemplate(RTETemplate):
    field_sep: str = "\n"

    @classmethod
    def get_template_name(cls):
        return "rte_gpt3_reproduce"


@dataclass
class WSCTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        tokens = sample["text"].split()
        txt1 = sample["target"]["span1_text"]
        txt2 = sample["target"]["span2_text"]
        idx1 = sample["target"]["span1_index"]
        idx2 = sample["target"]["span2_index"]
        # TODO This assertion fails because of incorrect tokenization
        # assert txt1.lower().startswith(tokens[idx1].lower()) and txt2.lower().startswith(tokens[idx2].lower())
        tokens[idx2] = "*" + txt2 + "*"  # Mark the pronoun in *bold*
        return self.train_sep.join(
            [
                " ".join(tokens),
                f'In the passage above, does the pronoun "*{txt2}*" refer to "{txt1}"? <mask>',
            ]
        )

    def verbalize(self, sample, candidate):
        return {"false": "No", "true": "Yes"}[candidate]


@dataclass
class CompositionalInstructionsClassificationCustomv1Template(FewShotTemplate):
    field_sep: str = "\n"

    full_task_verbalization_settings = {
        "task_prefix": JinjaTemplate(
            "Determine if a given text is {{ label_positive }} or {{ label_negative }}.\n"
        ),
        "task_summary": JinjaTemplate(
            "The text is considered {{ label_positive }} if it contains {{ positive_facts_summary }}. Follow these steps to verify that:\n"
        ),
        "task_ending": JinjaTemplate(
            "The text is {{ label_negative }} if any of the checks above is false.\n"
        ),
    }

    subtask_task_verbalization_settings = {
        "instruction": JinjaTemplate(
            "{{ instruction }} This is {{ label_positive }} when it is correct and {{ label_negative }} otherwise."
        )
    }

    def full_task_instruction(self, sample):
        verbalization_settings = self.full_task_verbalization_settings

        # Get the labels
        label_positive = sample.data["gold_label"]
        label_negative = [x for x in sample.data["labels"] if x != label_positive][0]

        flip_task_prefix = sample.data["source"].get("sample_id", 0) % 2 == 0
        task_prefix = verbalization_settings["task_prefix"].render(
            label_positive=label_negative if flip_task_prefix else label_positive, 
            label_negative=label_positive if flip_task_prefix else label_negative
        )  # "Determine if a given text is {{ label_positive }} or {{ label_negative}}.\n"

        # Get the facts summary wihout the template info (backward compatible with v0)
        # "task_summary": "The text is considered Label1 if it contains a word that does not look like a number and is not a digit. Follow these steps to verify that:"
        positive_facts_summary = sample.data["source"].get("positive_facts_summary_with_english_article", sample.data["source"]["task_summary"].replace("The text is considered Label1 if it contains ", "").replace(". Follow these steps to verify that:", ""))
        task_summary = verbalization_settings["task_summary"].render(
            label_positive=label_positive,
            label_negative=label_negative,
            positive_facts_summary=positive_facts_summary,
        )

        task_ending = verbalization_settings["task_ending"].render(
            label_positive=label_positive,
            label_negative=label_negative,
        ) # "The text is {{label_negative}} if any of the checks above is false.\n"

        task_sub_task_statments = [x["statement"] for x in sample.data["source"]["instruction_steps"]]
        task_lines = (
            [task_prefix, task_summary] + task_sub_task_statments + [task_ending]
        )
        full_task = "\n".join([x for x in task_lines if len(x) > 0])
        full_task = full_task.strip()

        return full_task

    def sub_task_instruction(self, sample):
        if "gold_label" not in sample.data:
            # this is called from the old task
            return sample["instruction"]

        verbalization_settings = self.subtask_task_verbalization_settings
        label_positive = sample.data["gold_label"]
        label_negative = [x for x in sample.data["labels"] if x != label_positive][0]

        instruction_new = verbalization_settings["instruction"].render(
            instruction=sample["instruction"],
            label_positive=label_positive,
            label_negative=label_negative
        )

        return instruction_new


    def encode(self, sample):
        if sample["full_task"]:
            instruction = self.full_task_instruction(sample)
        else:
            instruction = self.sub_task_instruction(sample)

        prompt = self.field_sep.join(
            [instruction, 
             "Text: ", 
             sample["input"], 
             "Output:",
             "<mask>"]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return candidate

    @classmethod
    def get_template_name(cls):
        return "compositional_instructions_classification_custom_v1"


@dataclass
class CompositionalInstructionsClassificationCustomv2NegatedTemplate(CompositionalInstructionsClassificationCustomv1Template):
    field_sep: str = "\n"

    full_task_verbalization_settings = {
        "task_prefix": JinjaTemplate(
            "Determine if a given text is {{ label_negative }} or {{ label_positive }}.\n"
        ),
        "task_summary": JinjaTemplate(
            "The text is considered {{ label_negative }} if it does not contain {{ positive_facts_summary }}. Follow these steps to verify that:\n"
        ),
        "task_ending": JinjaTemplate(
            "The text is {{ label_positive }} if all of the checks above are true.\n"
        ),
    }

    subtask_task_verbalization_settings = {
        "instruction": JinjaTemplate(
            "{{ instruction }} This is {{ label_negative }} when it is incorrect and {{ label_positive }} otherwise."
        )
    }

    @classmethod
    def get_template_name(cls):
        return "compositional_instructions_classification_custom_v2_negated"



@dataclass
class CompositionalInstructionsClassificationCustomv3SimpleTemplate(FewShotTemplate):
    """This is a template that is only one sentence"""
    field_sep: str = "\n"

    full_task_verbalization_settings = {
        "task_prefix": JinjaTemplate(
            ""
        ),
        "task_summary": JinjaTemplate(
            "The text is considered {{ label_positive }} if it contains {{ positive_facts_summary }}, otherwise it is {{ label_negative }}."
        ),
        "task_ending": JinjaTemplate(
            ""
        ),
    }

    def full_task_instruction(self, sample):
        verbalization_settings = self.full_task_verbalization_settings

        # Get the labels
        label_positive = sample.data["gold_label"]
        label_negative = [x for x in sample.data["labels"] if x != label_positive][0]

        flip_task_prefix = sample.data["source"].get("sample_id", 0) % 2 == 0
        task_prefix = verbalization_settings["task_prefix"].render(
            label_positive=label_negative if flip_task_prefix else label_positive, 
            label_negative=label_positive if flip_task_prefix else label_negative
        )  # "Determine if a given text is {{ label_positive }} or {{ label_negative}}.\n"

        # Get the facts summary wihout the template info (backward compatible with v0)
        # "task_summary": "The text is considered Label1 if it contains a word that does not look like a number and is not a digit. Follow these steps to verify that:"
        positive_facts_summary = sample.data["source"].get("positive_facts_summary_with_english_article", sample.data["source"]["task_summary"].replace("The text is considered Label1 if it contains ", "").replace(". Follow these steps to verify that:", ""))
        task_summary = verbalization_settings["task_summary"].render(
            label_positive=label_positive,
            label_negative=label_negative,
            positive_facts_summary=positive_facts_summary,
        )

        task_ending = verbalization_settings["task_ending"].render(
            label_positive=label_positive,
            label_negative=label_negative,
        ) # "The text is {{label_negative}} if any of the checks above is false.\n"

        task_sub_task_statments = [x["statement"] for x in sample.data["source"]["instruction_steps"]]
        task_lines = (
            [task_prefix, task_summary] 
            #+ task_sub_task_statments 
            + [task_ending]
        )
        task_lines = [x for x in task_lines if len(x.strip()) > 0]
        full_task = "\n".join([x for x in task_lines if len(x) > 0])
        full_task = full_task.strip()

        return full_task


    def encode(self, sample):
        instruction = self.full_task_instruction(sample)
        
        prompt = self.field_sep.join(
            [instruction, 
             "Input: ", 
             sample["input"], 
             "Output:",
             "<mask>"]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return candidate

    @classmethod
    def get_template_name(cls):
        return "cic_v3_with_complex_subtasks"


@dataclass
class CompositionalInstructionsClassificationCustomv4SimpleTemplate(CompositionalInstructionsClassificationCustomv3SimpleTemplate):
    """This is the template that Ves wanted as example"""
    field_sep: str = "\n"

    full_task_verbalization_settings = {
        "task_prefix": JinjaTemplate(
            ""
        ),
        "task_summary": JinjaTemplate(
            "Instructions:\nOutput {{ label_positive }} if the input contains {{ positive_facts_summary }}. Otherwise output {{ label_negative }}."
        ),
        "task_ending": JinjaTemplate(
            ""
        ),
    }

    @classmethod
    def get_template_name(cls):
        return "cic_v4_simple"
    
    
@dataclass
class CompositionalInstructionsClassificationCustomv4SimpleNegatedTemplate(CompositionalInstructionsClassificationCustomv3SimpleTemplate):
    """This is the template that Ves wanted as example but with a negation to counter the positional bias."""
    field_sep: str = "\n"

    full_task_verbalization_settings = {
        "task_prefix": JinjaTemplate(
            ""
        ),
        "task_summary": JinjaTemplate(
            "Instructions:\nOutput {{ label_negative }} if the input does not contain {{ positive_facts_summary }}. Otherwise output {{ label_positive }}."
        ),
        "task_ending": JinjaTemplate(
            ""
        ),
    }

    @classmethod
    def get_template_name(cls):
        return "cic_v4_simple_negated"
    
    
@dataclass
class CompositionalInstructionsClassificationCustomv4SimpleFLANTemplate(CompositionalInstructionsClassificationCustomv4SimpleTemplate):

    def encode(self, sample):
        task_description = self.full_task_instruction(sample)
        options = ["- "+cand for cand in sample.candidates]
        input = self.field_sep.join([sample["input"], "OPTIONS:"] + options)
        prompt = task_description + self.field_sep + self.field_sep + input + " <mask>"
        return prompt

    @classmethod
    def get_template_name(cls):
        return "cic_v4_simple_flanformat"

@dataclass
class CompositionalInstructionsClassificationCustomv4SimpleFLANNegatedTemplate(CompositionalInstructionsClassificationCustomv4SimpleNegatedTemplate):

    def encode(self, sample):
        task_description = self.full_task_instruction(sample)
        options = ["- "+cand for cand in sample.candidates]
        input = self.field_sep.join([sample["input"], "OPTIONS:"] + options)
        prompt = task_description + self.field_sep + self.field_sep + input + " <mask>"
        return prompt

    @classmethod
    def get_template_name(cls):
        return "cic_v4_simple_flanformat_negated"


@dataclass
class ReCoRDTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        text = sample["passage"]["text"]
        text = text.replace("@highlight", "")
        query = sample["qas"]["query"]
        query = query.replace("@placeholder", "<mask>")
        return self.train_sep.join([text, query])

    def verbalize(self, sample, candidate):
        start, end = candidate
        txt = sample["passage"]["text"]
        return txt[start:end]


@dataclass
class MultiRCTemplate(FewShotTemplate):
    train_sep: str = " "
    template_verbalize: Dict[str, str] = None  #
    template_prompt = "${context}\n${question}${answer_prompt}"
    template_answer_prompt = "\n- [${filler}] ${answer}"

    def verbalize(self, sample, candidate):
        if self.template_verbalize is None:
            self.template_verbalize = {"false": "False", "true": "True"}
        elif isinstance(self.template_verbalize, list):
            # This allows to define the tempalte_verbalize mapping as a list of tuples!
            self.template_verbalize = {k: v for k, v in self.template_verbalize}

        return self.template_verbalize[candidate]

    def encode(self, sample):
        assert not sample.has_subproblems

        prompt_template = Template(self.template_prompt)
        answer_prompt_template = Template(self.template_answer_prompt)

        encoded_prompt = prompt_template.substitute(
            context=sample["text"],
            question=sample["question"]["question"],
            answer_prompt=answer_prompt_template.substitute(
                answer=sample["question"]["answer"]["text"], filler="<mask>"
            ),
        )

        return encoded_prompt

    def encode_correct_candidate(self, sample):
        assert sample.has_subproblems

        prompt_template = Template(self.template_prompt)
        answer_prompt_template = Template(self.template_answer_prompt)

        answer_prompts_text = ""
        for subproblem in sample.subproblems:
            gold = self.verbalize(subproblem, subproblem.correct_candidates[0])
            answer_prompt = answer_prompt_template.substitute(
                answer=subproblem["question"]["answer"]["text"], filler=gold
            )
            answer_prompts_text += answer_prompt

        prompt = prompt_template.substitute(
            context=sample["text"],
            question=sample["question"]["question"],
            answer_prompt=answer_prompts_text,
        )

        return prompt


@dataclass
class MultiRCV1Template(FewShotTemplate):
    train_sep: str = " "
    template_verbalize: Dict[str, str] = None  #
    template_prompt = "$Context:{context}\nQuestion: ${question}${answer_prompt}"
    template_answer_prompt = "\nAnswer: ${answer} [{filler}]"


@dataclass
class MultiRCCorrectIncorrectTemplate(MultiRCTemplate):
    template_verbalize = [("true", "Correct"), ("false", "Incorrect")]

    @classmethod
    def get_template_name(cls):
        return "multirc_correct_incorrect"


@dataclass
class MultiRCRightWrongTemplate(MultiRCTemplate):
    template_verbalize = {"true": "Right", "false": "Wrong"}

    @classmethod
    def get_template_name(cls):
        return "multirc_right_wrong"


@dataclass
class MultiRCPositiveNegativeTemplate(MultiRCTemplate):
    template_verbalize = {"true": "Positive", "false": "Negative"}

    @classmethod
    def get_template_name(cls):
        return "multirc_positive_negative"


@dataclass
class MultiRCCorrectFalseTemplate(MultiRCTemplate):
    template_verbalize = {"true": "Correct", "false": "False"}

    @classmethod
    def get_template_name(cls):
        return "multirc_correct_false"


@dataclass
class MultiRCYesNoTemplate(MultiRCTemplate):
    template_verbalize = {"true": "Yes", "false": "No"}

    @classmethod
    def get_template_name(cls):
        return "multirc_yes_no"


@dataclass
class GPT3StyleNLITemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        prompt = self.train_sep.join(
            [
                f'Paragraph: {sample["sentence1"]}',
                f'Question: {sample["sentence2"]} True, False, or Neither?',
                "Answer: <mask>",
            ]
        )
        return prompt

    def verbalize(self, sample, candidate):
        return {"entailment": "True", "contradiction": "False", "neutral": "Neither"}[
            candidate
        ]


@dataclass
class PETStyleNLITemplate(FewShotTemplate):
    def encode(self, sample):
        sentence1 = sample["sentence1"].rstrip(".?!,;:'")
        sentence2 = sample["sentence2"]
        sentence2 = sentence2[0].lower() + sentence2[1:]  # Uncapitalize
        return f"{sentence1}? <mask>, {sentence2}"

    def verbalize(self, sample, candidate):
        return {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}[
            candidate
        ]


@dataclass
class DiagnosisTemplate(FewShotTemplate):
    def encode(self, sample):
        return ", ".join(sample["choices"]) + " -> <mask>"

    def verbalize(self, sample, candidate):
        return sample["choices"][candidate]


@dataclass
class SimplificationTemplate(FewShotTemplate):
    train_sep: str = " "
    separators: Optional[list] = field(
        default_factory=lambda: ["Paragraph:", "Question:", "Answer:"]
    )

    def encode(self, sample):
        source = sample["source"]
        prompt = self.train_sep.join(
            [
                f"Paragraph: {source}",
                "Question: Can you rephrase the above paragraph with simpler words to make it easier to read and understand by a child?",
                f"Answer: <mask>",
            ]
        )
        return prompt

    def postprocess(self, sample, candidate):
        if self.separators is None:
            return candidate
        separators_regex = "(:?" + "|".join(self.separators) + ")"
        return re.split(separators_regex, candidate)[0].strip()


@dataclass
class SyntheticTemplate(FewShotTemplate):

    def encode(self, sample):
        return sample["context"] + "<mask>"

    def postprocess(self, sample, candidate):
        return " " + candidate.strip().split(" ")[0].strip()


@dataclass
class UnscramblingTemplate(SyntheticTemplate):
    task_description: Optional[
        str
    ] = "Please unscramble the letters into a word, and write that word:"


@dataclass
class MTTemplate(FewShotTemplate):
    def encode(self, sample):
        return sample["src"] + " = <mask>"


@dataclass
class SATAnalogiesTemplate(FewShotTemplate):
    def encode(self, sample):
        word1 = sample["stem"]["word1"]
        word2 = sample["stem"]["word2"]
        return f"{word1} is to {word2} as <mask>"

    def verbalize(self, sample, candidate):
        word1 = sample["candidates"][candidate]["word1"]
        word2 = sample["candidates"][candidate]["word2"]
        return f"{word1} is to {word2}"


@dataclass
class PromptCompletionTemplate(FewShotTemplate):
    mask = " <mask>"

    def encode(self, sample):
        return sample["prompt_text"] + self.mask

    def postprocess(self, sample, candidate):
        return candidate


def get_values_hierarchical(dict_val, hier_field):
    field_hierarchy = [x.strip() for x in hier_field.split("->")]
    val = dict_val
    for fld in field_hierarchy:
        val = val[fld]

    return val


@dataclass
class NaturalInstructionsTemplate(FewShotTemplate):
    train_sep: str = " "
    field_sep: str = "\n"

    def encode(self, sample):
        instructions = sample["instructions"]
        instance = sample["instance"]

        instruction_fields_with_verbalization = [
            ("Title", "Title:"),
            ("Definition", "Definition:"),
            ("Things to Avoid", "Things to avoid:"),
            ("Emphasis & Caution", "Emphasis and caution:"),
            ("Examples->Positive Examples", "Positive examples:"),
            ("Examples->Negative Examples", "Negative examples:"),
            ("Prompt", "Prompt:"),
        ]

        instance_prompt = self.field_sep.join(
            ["Input: {0}".format(instance["input"]), "Output: <mask>"]
        )

        fields_to_join = [
            [field_verbal, instructions.get(field_key, None)]
            for field_key, field_verbal in instruction_fields_with_verbalization
        ]
        fields_to_join_non_empty = [
            f"{verbal} {value}" for verbal, value in fields_to_join if value is not None
        ]

        prompt = self.field_sep.join(fields_to_join_non_empty + [instance_prompt])
        return prompt

@dataclass
class FLANTemplate(FewShotTemplate):
    train_sep: str = " "
    def encode(self, sample):
      return sample.data["input"].replace(" X ", " <mask>")

    def verbalize(self, sample, candidate):
        return candidate # We are providing options in the input. So the verbalized version is the same as the candidate

@dataclass
class NaturalInstructionsExpansionTemplate(FewShotTemplate):
    train_sep: str = " "
    field_sep: str = "\n"
    no_template_input: bool = False
    def encode_instructions(self, sample, instruction_fields_with_verbalization):
        instructions = sample["instructions"]
        instance = sample["instance"]
        fields_to_join = []
        num_pos_example = 0
        num_neg_example = 0
        for field_key, field_verbal in instruction_fields_with_verbalization:
            if instructions.get(field_key, ""):
                value = instructions.get(field_key, "")
                if isinstance(value, str):
                    fields_to_join.append((field_verbal, " ".join(value.split()).strip()))
                elif isinstance(value, list):
                    for example_value in value:
                        if not isinstance(example_value, dict) or len(example_value['input']) == 0 or len(example_value['output']) == 0:
                            continue
                        example_str = [f"Input: {example_value['input']}", f"Output: {example_value['output']}"]
                        if 'explanation' in example_value:
                            example_str.append(f"Explanation: {example_value['explanation']}")
                        if field_verbal == "Positive example":
                            num_pos_example += 1
                            numbered_field_verbal = field_verbal + f"{num_pos_example}-"
                        elif field_verbal == "Negative example":
                            num_neg_example += 1
                            numbered_field_verbal = field_verbal + f"{num_neg_example}-"
                        else:
                            raise ValueError(f"unexpected field_verbal: {field_verbal}")
                        fields_to_join.append((numbered_field_verbal, self.field_sep.join(example_str).strip()))
                        
                else:
                    raise ValueError(f"unexpected type: {type(value)}")
        
        fields_to_join_non_empty  = [f"{verbal} {value}" if 'Positive example' not in verbal and 'Negative example' not in verbal else self.field_sep.join([verbal, value]) for verbal, value in fields_to_join]

        instance_prompt = self.field_sep.join(['Input: {0}'.format(instance["input"]),
                                               'Output: <mask>'])
        if self.no_template_input:
            prompt = self.field_sep.join(fields_to_join_non_empty)
        else:
            prompt = self.field_sep.join(fields_to_join_non_empty + [instance_prompt])
        return prompt

    def encode(self, sample):
        instruction_fields_with_verbalization = [
                              ('Definition', 'Definition:'), # only considering Definition for now
                              ('Positive Examples', 'Positive example'),
                              ('Negative Examples', 'Negative example'),
                              ]

        prompt = self.encode_instructions(sample, instruction_fields_with_verbalization)
        return prompt


@dataclass
class NIENoExampleTemplate(NaturalInstructionsExpansionTemplate):
        
    def encode(self, sample):
        instruction_fields_with_verbalization = [
                              ('Definition', 'Definition:'),
                              ]

        prompt = self.encode_instructions(sample, instruction_fields_with_verbalization)
        return prompt


@dataclass
class NIEOnlyExamplesTemplate(FewShotTemplate):
    train_sep: str = " "
    field_sep: str = "\n"

    def encode(self, sample):
        instructions = sample["instructions"]
        instance = sample["instance"]
        examples = instructions["Positive Examples"]
        examples_concat = []
        for ex in examples:
            examples_concat.append(f"Input: {ex['input']}")
            examples_concat.append(f"Output: {ex['output']}")
        examples_concat = self.field_sep.join(examples_concat)
        prompt = examples_concat + self.field_sep + f"Input: {instance['input']}" + self.field_sep + "Output: <mask>"
        return prompt


class NIENoExampleToFLANTemplate(FewShotTemplate):
    train_sep: str = " "
    field_sep: str = "\n"

    def encode(self, sample):
        instructions = sample["instructions"]
        instance = sample["instance"]
        task_description = instructions["Definition"]
        options = ["- "+cand for cand in sample.candidates]
        input = self.field_sep.join([instance["input"], "OPTIONS:"] + options)
        prompt = task_description + self.field_sep + self.field_sep + input + " <mask>"
        return prompt

    @classmethod
    def get_template_name(cls):
        return "nienoexample_to_flan"


@dataclass
class RegexTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        return sample["input"].replace("<mask>", "Answer: <mask>")


@dataclass
class RegexFLANTemplate(FewShotTemplate):
    train_sep: str = " "
    field_sep: str = "\n"

    def encode(self, sample):
        options = ["- "+cand for cand in sample.candidates]
        input = self.field_sep.join([sample["input"].replace("<mask>", "").strip(), "OPTIONS:"] + options)
        return input + " <mask>"

    @classmethod
    def get_template_name(cls):
        return "regex_flan"

        
@dataclass
class ProcessTextTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        prompt = f"<mask>"

        return prompt


@dataclass
class LAMATemplate(FewShotTemplate):
    def encode(self, sample):
        text_prompt = " ".join(sample["masked_sentences"])
        prompt = text_prompt.replace("[MASK]", "<mask>")
        return prompt


@dataclass
class LAMAGoogleRETemplate(FewShotTemplate):
    def encode(self, sample):
        relation = sample["pred"]
        if "place_of_birth" in relation:
            template = "[X] was born in [Y] ."
        elif "date_of_birth" in relation:
            template = "[X] (born [Y])."
        elif "place_of_death":
            template = "[X] died in [Y] ."
        else:
            raise Exception("Unknown relation!")
        prompt = template.replace("[X]", sample["sub_label"]).replace("[Y]", "<mask>")
        return prompt


@dataclass
class LAMATRExTemplate(FewShotTemplate):
    def encode(self, sample):
        template = sample["template"]
        prompt = template.replace("[X]", sample["sub_label"]).replace("[Y]", "<mask>")
        return prompt


@dataclass
class MLAMATemplate(FewShotTemplate):
    def encode(self, sample):
        template = sample["template"]
        prompt = template.replace("[X]", sample["sub_label"]).replace("[Y]", "<mask>")
        return prompt


@dataclass
class BlimpTemplate(FewShotTemplate):
    def encode(self, sample):
        return "<mask>"

    def verbalize(self, sample, candidate):
        return sample[candidate]


if __name__ == "__main__":
    for template in sorted(get_all_templates()):
        print(template)
