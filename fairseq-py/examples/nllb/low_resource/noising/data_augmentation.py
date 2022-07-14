"""
Implements various noising functions:
- language modeling 
- masking
- shuffling
- deletion

Example usage:
python examples/nllb/low_resource/noising/data_augmentation.py --basepath /private/home/angelafan/nllb/fairseq-py/process_holger_bitexts/eng-zul --source eng --target zul --prob 0.3 --testing
"""
import random
import math 
import numpy as np
import argparse
import subprocess


def main(args):
    if args.testing:
        source_sentence = "we are testing these noising functions"
        target_sentence = "nous testons ces fonctions de bruit"

        print(mask(source_sentence))
        print(shuffle(source_sentence))
        print(deletion(source_sentence))
        print(apply_all_functions(source_sentence, target_sentence, args.source, args.target))
    else:
        BASE=args.basepath

        if args.prob > 1 or args.prob < 0:
            assert("Error, probability must be between 0 and 1")

        # training dataset
        inputs, outputs = build_corpus(f"{BASE}/filter.{args.source}", f"{BASE}/filter.{args.target}", args.source, args.target, args.prob)

        with open(f"{BASE}/filter_noise.{args.source}", "w") as o:
            for line in inputs: 
                o.write(line.strip() + "\n")

        with open(f"{BASE}/filter_noise.{args.target}", "w") as o:
            for line in outputs: 
                o.write(line.strip() + "\n")

        # evaluation dataset
        for split in ["devtest", "dev"]:
            evaluation_source = apply_tags_evaluation(BASE, split, args.source)
            with open(f"{BASE}/{split}noise.{args.source}", "w") as o:
                for line in evaluation_source:
                    o.write(line.strip() + "\n")
            execute_in_shell(f"cp {BASE}/tok.{split}.{args.target} {BASE}/{split}noise.{args.target}")

        print(f"Finished writing {BASE}/noise.{args.target} and {BASE}/noise.{args.source}")
        execute_in_shell(f"head {BASE}/*noise*")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", help="base path where data is located")
    parser.add_argument("--source", help="source lang")
    parser.add_argument("--target", help="target lang")
    parser.add_argument("--prob", help="probability of masking/deletion/shuffling", type=float)
    parser.add_argument("--testing", help="testing only", default=False, action="store_true")
    return parser.parse_args()

def execute_in_shell(command):
    print(command)
    subprocess.call(command, shell=True)

def mask(sequence, prob=0.3):
    # returns the sequence with a random number of tokens replaced with "mask"
    sequence = sequence.split()
    samples = set(random.sample(range(len(sequence)), math.ceil(prob*len(sequence))))
    masked_sequence = ""
    for i, word in enumerate(sequence):
        if i in samples:
            masked_sequence += "[mask] "
        else:
            masked_sequence += f"{word} "
    return masked_sequence.strip()

def shuffle(sequence):
    # returns the sequence in a shuffled order 
    sequence = sequence.split()
    random.shuffle(sequence)
    return " ".join(sequence)

def deletion(sequence, prob=0.3):
    # returns the sequence with a random number of deleted elements
    sequence = sequence.split()
    samples = set(random.sample(range(len(sequence)), math.ceil(prob*len(sequence))))
    deleted_sequence = ""
    for i, word in enumerate(sequence):
        if i not in samples:
            deleted_sequence += f"{word} "
    return deleted_sequence.strip()

def apply_all_functions(source, target, sourcelang, targetlang, prob=0.3):
    """
    Compose the noising functions so they are applied to create more difficult translation situations
    """     
    inputs = []
    outputs = []
    new_input = mask(source, prob=prob)
    inputs.append(f"masking_task {sourcelang} {new_input}")
    outputs.append(target)
    # mask target, predict target
    new_input = mask(target, prob=prob)
    inputs.append(f"masking_task {targetlang} {new_input}")
    outputs.append(target)
    # shuffle source, predict target 
    new_input = shuffle(source)
    inputs.append(f"shuffling_task {sourcelang} {new_input}")
    outputs.append(target)
    # shuffle target, predict target 
    new_input = shuffle(target)
    inputs.append(f"shuffling_task {targetlang} {new_input}")
    outputs.append(target)
    # delete source, predict target 
    new_input = deletion(source, prob=prob)
    inputs.append(f"deletion_task {sourcelang} {new_input}")
    outputs.append(target)
    # delete target, predict target 
    new_input = deletion(target, prob=prob)
    inputs.append(f"deletion_task {targetlang} {new_input}")
    outputs.append(target)

    # add the real task
    inputs.append(f"translation_task {sourcelang} {source}")
    outputs.append(target)

    assert(len(inputs) == len(outputs))
    return inputs, outputs

def build_corpus(source_file, target_file, sourcelang, targetlang, prob=0.3):
    """
    Apply 'apply_all_functions' to all lines in a dataset
    """
    with open(source_file) as f:
        source_data = f.readlines()
    with open(target_file) as f:
        target_data = f.readlines()
    assert(len(source_data) == len(target_data))
    inputs = []
    outputs = []
    for i, source_line in enumerate(source_data):
        source_line = source_line.strip()
        target_line = target_data[i].strip()
        if filter(source_line, target_line):
            new_input, new_target = apply_all_functions(source_line, target_line, sourcelang, targetlang, prob=prob)
            inputs.extend(new_input)
            outputs.extend(new_target)
    assert(len(inputs) == len(outputs))
    # deduplicate
    inputs, outputs = zip(*set(zip(inputs, outputs)))
    return list(inputs), list(outputs)

def filter(sentence1, sentence2):
    """
    Remove sentences if they are shorter than 5 SPM units or contain less than 20 characters
    """
    if len(sentence1.split()) < 5 or len(sentence2.split()) < 5:
        return False 
    if len(sentence1) < 20 or len(sentence2) < 20:
        return False 
    return True

def apply_tags_evaluation(basepath, split, sourcelang):
    """
    Apply frontmatter tags to valid and test data
    """
    with open(f"{basepath}/tok.{split}.{sourcelang}") as f:
        source_data = f.readlines()
    new_source = [f"translation_task {sourcelang} {source_sentence}" for source_sentence in source_data]
    assert(len(source_data) == len(new_source))

    return new_source


if __name__ == "__main__":
    args = parse_args()
    main(args)