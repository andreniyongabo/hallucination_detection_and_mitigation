#!/usr/bin/env python3


import csv
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import mwparserfromhell
from spacy.lang.en import English

# Spacy is used for sentence splitting.
nlp = English()
nlp.add_pipe("sentencizer")

ns = {"": "http://www.mediawiki.org/xml/export-0.10/"}
tree = ET.parse("raw/wikipedia_dump.xml")
root = tree.getroot()

articles = {}

# List of English wikipedia titles (and broad categories) considered fundamental.
category_to_articles = defaultdict(set)
article_to_category = {}
with open("global_list.tsv", "rt") as f:
    for category, article in csv.reader(f, delimiter="\t"):
        category_to_articles[category].add(article)
        article_to_category[article] = category
num_categories = len(category_to_articles)
print(
    f"Read list of {len(article_to_category)} articles in {num_categories} categories."
)

# Some regex patterns of markup that the parser consistently fails to clean up, and
# that we'll have to remove manually.
refs = re.compile(r"<ref>.+?</ref>")
full_line_template = re.compile(r"^\s*\[\[.+\]\]\s*$", re.M)
basic_templates = re.compile(r"\[\[(File|Category):[^\]]+\]\]")


def cleanup_wikicode(wikicode):
    wikicode = re.sub(refs, "", wikicode)
    wikicode = re.sub(full_line_template, "", wikicode)
    wikicode = re.sub(basic_templates, "", wikicode)
    return wikicode


# On average, sections with these headings were observed to contain mostly long lists
# of wikilinks, and not much useful language. We'll clean them up.
bad_sections = re.compile(
    r"^\s*(Awards|Awards and honors|Awards and nominations|Awards, honors and tributes|Bibliography|Books|Citations|Discography|Documentaries|External links|Features|Filmography|Further reading|General sources|Interviews|Notes|Portrayals|Radio appearances|References|References and notes|See also|Stage|Television)\s*$.*",
    re.DOTALL | re.M | re.I,
)


def cleanup_parser_output(text):
    return re.sub(bad_sections, "", text)


# Some simple heuristics to get rid of sentences that don't contain real language,
# mostly wikicode markup that the parser failed to remove, and bibliographic references
# that the parser mistook for body text.
def is_good_sentence(sent):
    tokens_ok = 5 <= sent.count(" ") + 1 <= 60
    no_bad_substrings = all(
        c not in sent
        for c in [
            " p.",  # leftover from references
            " pp.",  # leftover from references
            "/",
            "|",
            "[",
            "]",
            "{",
            "}",
            "*",
            "\n",
            "\t",
            " ,",  # typically indicates a foreign template
            " ;",
            "  ",
        ]
    )
    start_ok = not sent.startswith(")")
    end_ok = not sent.endswith("(")
    return tokens_ok and no_bad_substrings and start_ok and end_ok


# The parser often fails to properly parse templates involving dates, and days end up
# getting stuck to months without a space. This can be largely fixed automatically.
stuck_dates_1 = re.compile(
    r"([0-9])(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.I
)
stuck_dates_2 = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December) ([0-9]{4})([0-9])",
    re.I,
)
bad_parens = re.compile(r" \(([,;]|\s|or|and|away|deep|long|BC)*\)")
space_before_date_in_parens = re.compile(r"\(\s([0-9]{2,})")


def cleanup_sent(s):
    s = s.strip()
    s = re.sub(stuck_dates_1, r"\1 – \2", s)
    s = re.sub(stuck_dates_2, r"\1 \2 \3", s)
    s = re.sub(bad_parens, "", s)
    s = re.sub(space_before_date_in_parens, r"(\1", s)
    return s


# From each page, extracts every triplet of contiguous sentences that didn't trigger
# any of the heuristic defined above. These will need to be sampled further.
with open("sentences.tsv", "wt") as f, open("sentences.err", "wt") as ferr:
    writer = csv.writer(f, delimiter="\t")
    for i, page in enumerate(root.findall("page", ns)):
        title = page.find("title", ns).text
        print(f"{i}/{len(article_to_category)}: {title}")
        try:
            category = article_to_category[title]
        except KeyError:
            ferr.writer(f"Skipping unknown article: {title}.\n")
            pass
        wikicode = page.find("revision", ns).find("text", ns).text
        wikicode = cleanup_wikicode(wikicode)
        article = cleanup_parser_output(mwparserfromhell.parse(wikicode).strip_code())
        sentences = list(cleanup_sent(s.text) for s in nlp(article).sents)
        contiguous = [
            sentences[n - 3 : n]
            for n in range(3, len(sentences), 3)
            if is_good_sentence(sentences[n - 1])
            and is_good_sentence(sentences[n - 2])
            and is_good_sentence(sentences[n - 3])
        ]
        if len(contiguous) < 1:
            ferr.write(f"Couldn't find three good contiguous sentences in: {title}.\n",)
            continue
        for sent1, sent2, sent3 in contiguous:
            writer.writerow([category, title, sent1, sent2, sent3])
