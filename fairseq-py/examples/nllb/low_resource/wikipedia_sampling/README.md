# Sampling sentences from Wikipedia

The scripts in this folder are for sampling broad-coverage sentences from Wikipedia,
for the purposes of obtaining source data to send off to translators for seed data
collection.

The whole approach is centered around Wikimedia's [List of articles every Wikipedia should have](https://meta.wikimedia.org/wiki/List_of_articles_every_Wikipedia_should_have/Expanded),
a collection of 10,000 Wikidata IDs corresponding to the most notable topics in
different fields of knowledge and human activity. These are split into 11 categories,
such as `People`, `History`, `Georgraphy`, `Arts`, `Philosophy and religion`, etc.

The script should be run in numerical order:
* `01-download_data.sh` downloads the list above as well as dumps of Wikidata and
  Wikipedia.
* `02-build_wikidata_mapping.py` uses the Wikidata dump to create a mapping to convert
  between Wikidata IDs and English Wikipedia page titles.
* `03-extract_list.py` reads the list of Wikidata IDs, extracts them, and converts them
  to Wikipedia page titles.
* `04-get_sentence_triplets.py` extracts triplets of contiguous sentences from the
  Wikipedia pages above; parses wikicode into plain text; additionally uses heuristics
  to detect the sentences that were not fully parsed by the parser and still contain
  wikicode, or other common parsing problems. Note: parsing wikicode is a nightmare,
  as might be guessed from the name of the parser (`mwparserfromhell`). Many templates,
  such as those for rendering dates, often fail to parse correctly. It's not a bad idea
  to spend 15-20 minutes visually inspecting the output from this script and manually
  deleting any lines that look too crazy.
* `05-sample_10k_sents.py` samples 10k sentences from those extracted in the previuos
  step, such that the mean sentence length is close to 25 words; it then outputs a TSV
  file with 3,333 triplets (~10k sentences), as well as some extra sentences to be used
  as replacements should any of the original ones be rejected by translators. This
  ensures we have a disjoint set of potential replacement sentences that don't overlap.
