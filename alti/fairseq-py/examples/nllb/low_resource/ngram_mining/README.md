# Translation-based mining

##### 1. Tokenization
Apply tokenization (NLTK tokenizer) to:
- English monolingual sentences
- English-translated foreign sentences
Support sharding for parellization

Example:
```
bash examples/nllb/low_resource/ngram_mining/scripts/run_tokenize.sh
```

##### 2. Creating inverted index
For each tokenized sentence, generate a list n-grams for the sentence. Create an inverted index from n-gram to sentence ids (for all the sentences containing that n-gram)

We use map-reduce for parallelization.
* Map-step: For each input tokenized file, create inverted indexes sharded by n-gram
* Reduce-step: Combine sharded-ngrams indexes from all input files

Example
```
bash examples/nllb/low_resource/ngram_mining/scripts/run_create_index_map.sh
bash examples/nllb/low_resource/ngram_mining/scripts/run_create_index_reduce.sh
```

##### 3. Mining
From the n-gram indexes, find sentences that share overlapping n-grams. Filter to only non-popular n-grams (n-grams that appear in fewer than X sentences). Compute scores (BLEU/TER) for the candidate pairs

Example
```
bash examples/nllb/low_resource/ngram_mining/scripts/run_ngram_mine.sh
```
##### 4. Filtering
Combine mining results from all shards, and filter scores based on threshold

Example
```
python examples/nllb/low_resource/ngram_mining/scripts/filter.py
```
