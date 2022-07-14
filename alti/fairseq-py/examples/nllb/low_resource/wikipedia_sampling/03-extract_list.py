#!/usr/bin/env python3

import csv
import re

from scrapy.http import HtmlResponse


# Extract links from bullet point lists.
def parse_global(html):
    response = HtmlResponse("", body=html)
    return response.css("li a[href*='wikidata.org/wiki/Q']::attr(href)").getall()


# Mapping from Wikidata IDs to the corresponding enwiki page titles.
with open("wikidata_to_enwiki.tsv", "rt") as f:
    wikidata_to_enwiki = {
        wikidata: enwiki for wikidata, enwiki in csv.reader(f, delimiter="\t")
    }

# List of sections from the following page:
# https://meta.wikimedia.org/wiki/List_of_articles_every_Wikipedia_should_have/Expanded
PAGES = "People History Geography Arts Philosophy_and_religion Anthropology,_psychology_and_everyday_life Society_and_social_sciences Biology_and_health_sciences Physical_sciences Technology Mathematics".split()
WIKIDATA_LINK = re.compile("https://www\.wikidata\.org/wiki/(Q[0-9]+)$")

# Extract of English Wikipedia page titles corresponding to pages which are considered
# fundamental. Write them to a TSV file, including their broad category too.
count = 0
skipped = 1
with open("global_list.tsv", "wt") as fout:
    for page in PAGES:
        category = page.replace("_", " ")
        with open(f"html/Global:{page}.html", "rb") as fin:
            for link in parse_global(fin.read()):
                match = re.match(WIKIDATA_LINK, link)
                if match:
                    entity_id = match.group(1)
                    if entity_id not in wikidata_to_enwiki:
                        skipped += 1
                    else:
                        fout.write(f"{category}\t{wikidata_to_enwiki[entity_id]}\n")
                        count += 1
