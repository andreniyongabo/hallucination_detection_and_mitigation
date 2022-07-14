#!python3 -u

import argparse
import glob
import os
import re
import shutil
import unittest


class TestCase(unittest.TestCase):
    def test_normalize_lang(self):
        self.assertEqual(normalize_langname("zM-tfNG"), "tzm_Tfng")
        self.assertEqual(normalize_langname("ace"), "ace")
        self.assertEqual(normalize_langname("aCe"), "ace")
        self.assertEqual(normalize_langname("aCe-lAtN"), "ace_Latn")
        self.assertEqual(normalize_langname("aCe_lAtN"), "ace_Latn")
        self.assertEqual(normalize_langname("aCe_lAtN_ma"), "ace_Latn_MA")


def normalize_langname(lang_name):
    if "-" in lang_name:
        parts = lang_name.split("-")
    elif "_" in lang_name:
        parts = lang_name.split("_")
    else:
        parts = [lang_name]

    # Normalize naming
    parts = [x.lower() for x in parts]
    if len(parts) > 1:
        parts[1] = parts[1].capitalize()
    if len(parts) > 2:
        parts[2] = parts[2].upper()

    val = "_".join(parts)

    if val == "zm_Tfng":  # fix file naming consistency
        val = "tzm_Tfng"
    elif val == "tzm_Tfng_MA":  # fix file naming consistency
        val = "tzm_Tfng"
    return val


def extract_files(files):
    prog = re.compile("(.*)_sentences_(\d*).tsv")
    file_by_lang = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        m = prog.match(filename)
        if m:
            lang_name = m.group(1)
            lang_name = normalize_langname(lang_name)
            if not lang_name in file_by_lang:
                file_by_lang[lang_name] = []
            file_by_lang[lang_name].append(file_path)

    return file_by_lang


def extract_folder(folder_name, output_folder):
    files = glob.glob(folder_name + "/**/*.tsv", recursive=True)
    file_by_lang = extract_files(files)
    os.makedirs(output_folder, exist_ok=True)

    for lang_name, files in file_by_lang.items():
        output_file_name = f"{lang_name}.tsv"
        output_file = os.path.join(output_folder, output_file_name)
        print(output_file)
        with open(output_file, "wb") as f_w:
            for in_file in files:
                with open(in_file, "rb") as f_r:
                    shutil.copyfileobj(f_r, f_w)


def main():
    parser = argparse.ArgumentParser(
        description="This script aggregates translated `.tsv` files per language"
    )
    parser.add_argument("--vendors-folder", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    extract_folder(args.vendors_folder, args.output)


if __name__ == "__main__":
    main()
