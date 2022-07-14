#!/usr/bin/env python3


import csv
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

ARTICLES_PER_CATEGORY = 500


@dataclass
class Triplet:
    category: str
    title: str
    s1: str
    s2: str
    s3: str
    words: int

    def __init__(self, category, title, s1, s2, s3):
        self.category = category
        self.title = title
        self.s1, self.s2, self.s3 = s1, s2, s3
        self.words = 3 + s1.count(" ") + s2.count(" ") + s3.count(" ")


collection = defaultdict(lambda: defaultdict(list))
with open("sentences.tsv", "rt") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, row in enumerate(reader):
        try:
            category, title, s1, s2, s3 = row
        except ValueError:
            print(row)
            raise
        collection[category][title].append(Triplet(category, title, s1, s2, s3))

print(f"Read {i+1} sentence triplets.")

category_names = list(collection.keys())
assert len(category_names) == 11, category_names

article_titles = {
    category: list(articles.keys()) for category, articles in collection.items()
}

# We want to sample 10k sentences in such a way that the mean sentence length is as
# close as possible to 25 words. Naive solution, since it's fast enough: repeat the
# sampling N times, pick the set of samples that's closest to a mean of 25.
samples = []
for _ in range(100):
    sample = {}
    total_num_words = 0
    total_num_triplets = 0
    for category in category_names:
        sampled_article_titles = random.sample(
            article_titles[category],
            k=min(ARTICLES_PER_CATEGORY, len(article_titles[category])),
        )
        sampled_triplets = [
            random.choice(collection[category][title])
            for title in sampled_article_titles
        ]
        total_num_triplets += len(sampled_triplets)
        total_num_words += sum(triplet.words for triplet in sampled_triplets)
        sample[category] = sampled_triplets
    avg_words_per_sentence = total_num_words / (3 * total_num_triplets)
    samples.append((avg_words_per_sentence, sample))

# Select the sample that's closest to a mean length of 25 words
samples = sorted(samples, key=lambda s: abs(s[0] - 25))
sample = samples[0][1]
sample = [triplet for category in sample.values() for triplet in category]
random.shuffle(sample)  # shuffle the sentences in the sample

# We split up our sample. We're only interested in 10k sentences, but we will still
# keep around any extra sentences in case any of the data is rejected by vendors and
# we need replacement sentences.
sample, other_sample = sample[:3333], sample[3333:]
sample = sorted(sample, key=lambda t: (t.category, t.title))
lengths = [t.words for t in sample]
avg_length = np.average(lengths) / 3
stdev = np.sqrt(np.var(lengths) / 3)
with open("sampled_sentences.tsv", "wt") as f:
    writer = csv.writer(f, delimiter="\t")
    for triplet in sample:
        writer.writerow((triplet.title, triplet.s1))
        writer.writerow((triplet.title, triplet.s2))
        writer.writerow((triplet.title, triplet.s3))
print(f"Saved {len(sample)*3} sentences, µ={avg_length}, σ={stdev}")

other_sample = sorted(other_sample, key=lambda t: (t.category, t.title))
lengths = [t.words for t in other_sample]
avg_length = np.average(lengths) / 3
stdev = np.sqrt(np.var(lengths) / 3)
with open("other_sampled_sentences.tsv", "wt") as f:
    writer = csv.writer(f, delimiter="\t")
    for triplet in other_sample:
        writer.writerow((triplet.title, triplet.s1))
        writer.writerow((triplet.title, triplet.s2))
        writer.writerow((triplet.title, triplet.s3))
print(f"Saved {len(other_sample)*3} extra sentences, µ={avg_length}, σ={stdev}")
