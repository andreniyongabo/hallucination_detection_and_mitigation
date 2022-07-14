#!/usr/bin/env python
# -*- coding: utf-8 -*-


from langdetect import detect
import sys


langdetect_to_iso_639_3 = {'af': 'afr', 'sq': 'sqi', 'ar': 'ara', 'bn': 'ben', 'no': 'nor', 'bg': 'bul', 'ca': 'cat', 'zh-cn': 'zho', 'hr': 'hrv', 'cs': 'ces', 'da': 'dan', 'nl': 'nld', 'en': 'eng', 'et': 'est', 'fi': 'fin', 'fr': 'fra', 'de': 'deu', 'gu': 'guj', 'he': 'heb', 'hi': 'hin', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'ja': 'jpn', 'kn': 'kan', 'ko': 'kor', 'lv': 'lav', 'lt': 'lit', 'mk': 'mkd', 'ml': 'mal', 'mr': 'mar', 'el': 'ell', 'ne': 'npi', 'fa': 'fas', 'pl': 'pol', 'pt': 'por', 'pa': 'pan', 'ro': 'ron', 'ru': 'rus', 'sk': 'slk', 'sl': 'slv', 'so': 'som', 'es': 'spa', 'sw': 'swh', 'sv': 'swe', 'tl': 'tgl', 'ta': 'tam', 'te': 'tel', 'th': 'tha', 'tr': 'tur', 'uk': 'ukr', 'ur': 'urd', 'vi': 'vie', 'cy': 'cym'}



def main():
    ctr = 0
    for line in sys.stdin:
        if line[-1] == "\n":
            line = line[:-1]
        gold_label = line.split()[0]
        sentence = line[len(gold_label)+1:]

        prediction_language = detect(sentence)
        prediction_language_iso_639_3 = langdetect_to_iso_639_3.get(prediction_language, "unk")

        print(f"__label__{prediction_language_iso_639_3}")


if __name__ == "__main__":
    main()
