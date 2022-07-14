#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cld3
import sys

cld3_to_iso_639_3 = {'eu': 'eus', 'eo': 'epo', 'ny': 'nya', 'haw': 'haw', 'sm': 'smo', 'st': 'sot', 'af': 'afr', 'sq': 'sqi', 'am': 'amh', 'ar': 'ara', 'hy': 'hye', 'az': 'aze', 'bn': 'ben', 'no': 'nor', 'bs': 'bos', 'bg': 'bul', 'my': 'mya', 'ca': 'cat', 'ceb': 'ceb', 'zh': 'zho', 'hr': 'hrv', 'cs': 'ces', 'da': 'dan', 'nl': 'nld', 'en': 'eng', 'et': 'est', 'fi': 'fin', 'fr': 'fra', 'gl': 'glg', 'ka': 'kat', 'de': 'deu', 'gu': 'guj', 'ht': 'hat', 'ha': 'hau', 'iw': 'heb', 'hi': 'hin', 'hu': 'hun', 'is': 'isl', 'ig': 'ibo', 'id': 'ind', 'ga': 'gle', 'it': 'ita', 'ja': 'jpn', 'jv': 'jav', 'kn': 'kan', 'kk': 'kaz', 'km': 'khm', 'ko': 'kor', 'ku': 'kur', 'ky': 'kir', 'lo': 'lao', 'lv': 'lav', 'lt': 'lit', 'lb': 'ltz', 'mk': 'mkd', 'mg': 'mlg', 'ms': 'msa', 'ml': 'mal', 'mt': 'mlt', 'mi': 'mri', 'mr': 'mar', 'el': 'ell', 'mn': 'mon', 'ne': 'npi', 'ps': 'pus', 'fa': 'fas', 'pl': 'pol', 'pt': 'por', 'pa': 'pan', 'ro': 'ron', 'ru': 'rus', 'gd': 'gla', 'sr': 'srp', 'sn': 'sna', 'sd': 'snd', 'si': 'sin', 'sk': 'slk', 'sl': 'slv', 'so': 'som', 'es': 'spa', 'su': 'sun', 'sw': 'swh', 'sv': 'swe', 'fil': 'tgl', 'tg': 'tgk', 'ta': 'tam', 'te': 'tel', 'th': 'tha', 'tr': 'tur', 'uk': 'ukr', 'ur': 'urd', 'uz': 'uzb', 'vi': 'vie', 'cy': 'cym', 'xh': 'xho', 'yi': 'yid', 'yo': 'yor', 'zu': 'zul', 'be': 'bel', 'co': 'cos', 'fy': 'fry'}


def main():
    ctr = 0
    for line in sys.stdin:
        if line[-1] == "\n":
            line = line[:-1]
        gold_label = line.split()[0]
        sentence = line[len(gold_label)+1:]

        prediction = cld3.get_language(sentence)
        if not hasattr(prediction, "language"): # usually an empty line
            print(f"__label__unk")
        else:
            prediction_language = prediction.language
            if prediction_language not in cld3_to_iso_639_3:
                print(f"__label__cld3_{prediction_language}")
            else:
                prediction_language_iso_639_3 = cld3_to_iso_639_3[prediction_language]
                print(f"__label__{prediction_language_iso_639_3}")


if __name__ == "__main__":
    main()
