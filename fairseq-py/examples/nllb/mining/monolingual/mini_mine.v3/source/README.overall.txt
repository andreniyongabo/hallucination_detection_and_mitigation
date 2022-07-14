Processing code to turn monolingual data into clean monolingual data, usable for mining, backtranslation, and more.
Runs the following:
sentence segmentation based on either existing tokenizers, rules, or backup to the closest language with script support (using language_equivalences.tsv)
filters data with basic heuristics
filters data based on script (in language_scripts_200.tsv)
performs language identification
deduplicates

Example:
bash process.sh

Dependencies:
pip install indic-nlp-library
pip install pythainlp
pip install laonlp
pip install khmer-nltk
pip install laonlp
pip install python-crfsuite
