#!/usr/bin/env python3
import gzip

import ijson

# Create a mapping between Wikidata IDs and the corresponding enwiki page titles.
with gzip.open("raw/wikidata.json.gz", "r") as fin, open(
    "wikidata_to_enwiki.tsv", "wt"
) as fout:
    for item in ijson.items(fin, "item"):
        try:
            out = item["id"] + "\t" + item["sitelinks"]["enwiki"]["title"] + "\n"
            fout.write(out)
        except KeyError:
            pass
