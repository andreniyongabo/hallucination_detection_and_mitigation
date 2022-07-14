#!/usr/bin/env python

import argparse
import os
import re
import sys

import threading
import time


# Based on https://www.internalfb.com/intern/anp/view/?id=1215095&checkpoint_id=1023178205186735


arab_char = re.compile(r"^06[0-9].|^06[A-F].|^08[A-F].")
armn_char = re.compile(r"^05[3-9].")
bali_char = re.compile(r"^1B[0-7][0-9]|^1B[0-7][A-F]")
beng_char = re.compile(r"^09[8-9].|^09[A-F].")
bugi_char = re.compile(r"^1A[0-1].")
cans_char = re.compile(r"^18[B-F].")
cyrl_char = re.compile(r"^04[0-9].|^04[A-F].|^05[1-2].")
deva_char = re.compile(r"^09[0-7].")
ethi_char = re.compile(r"^12[0-9].|^12[A-F].")
geor_char = re.compile(r"^10[A-F].|^1C9.|^1C[A-B].")
grek_char = re.compile(r"^03[7-9].|^03[A-F].|^1F[0-9].|^1F[A-F].")
gujr_char = re.compile(r"^0A[8-9].|^0A[A-F].")
guru_char = re.compile(r"^0A[0-7].")
han_char = re.compile(r"^33[0-9].|^33[E-F].|^34[0-9].|^34[A-F].|^4E[0-9].|^4E[A-F].|^68[0-9].|^6B[A-F].|^75[0-9].|^7B[0-9].")
hang_char = re.compile(r"^11[0-9].|^FF[A-D].")
hebr_char = re.compile(r"^059.|^05[A-F].")
java_char = re.compile(r"^A9[8-9].|^A9[A-D].")
jpan_char = re.compile(r"^30[4-9].|^30[A-F].|^FF[6-9].")
khmr_char = re.compile(r"^17[8-9].|^17[A-F].|^19[E-F].")
knda_char = re.compile(r"^0C[8-9].|0C[A-F].")
laoo_char = re.compile(r"^0E[8-9].|^0E[A-E].")
latn_char = re.compile(r"^00[0-7].|^00[A-F].|^01[0-7].|^0254|^025B|^0272")
marc_char = re.compile(r"^11C[7-9].|^11C[A-B].")
mtei_char = re.compile(r"^AB[C-E].|^ABF[0-9]|^AAE.|^AAF[0-6]")
mlym_char = re.compile(r"^0D[0-7].")
mong_char = re.compile(r"^18[0-9].|^18A.")
mymr_char = re.compile(r"^10[0-9].")
nkoo_char = re.compile(r"^07[C-F].")
olck_char = re.compile(r"^1C[5-7][0-9]|^1C[5-7][A-F]")
phag_char = re.compile(r"^A8[4-7].")
orya_char = re.compile(r"^0B[0-7].")
sinh_char = re.compile(r"^0D[8-9].|^0D[A-F].")
sund_char = re.compile(r"^1B[8-9].|^1B[A-B].|^1CC.")
taml_char = re.compile(r"^0B[8-9].|^0B[A-F].")
telu_char = re.compile(r"^0C[0-7].")
tfng_char = re.compile(r"^2D[3-7].")
thai_char = re.compile(r"^0E[0-7].")
tibt_char = re.compile(r"^0F[0-9].|^0F[A-F].")


def main():
    parser = argparse.ArgumentParser(description="Predict language script from the stdin")
    parser.add_argument("--filter-mode", action="store_true", help="show the original text")
    args = parser.parse_args()

    pattern = re.compile(r"[\s,;:?!@&]")

    regex_map = {
        "Arab":  arab_char,
        "Armn":  armn_char,
        "Bali":  bali_char,
        "Beng":  beng_char,
        "Bugi":  bugi_char,
        "Cans":  cans_char,
        "Cyrl":  cyrl_char,
        "Deva":  deva_char,
        "Ethi":  ethi_char,
        "Geor":  geor_char,
        "Grek":  grek_char,
        "Gujr":  gujr_char,
        "Guru":  guru_char,
        "Han" :  han_char,
        "Hang":  hang_char,
        "Hebr":  hebr_char,
        "Java":  java_char,
        "Jpan":  jpan_char,
        "Khmr":  khmr_char,
        "Knda":  knda_char,
        "Laoo":  laoo_char,
        "Latn":  latn_char,
        "Marc":  marc_char,
        "Mlym":  mlym_char,
        "Mong":  mong_char,
        "Mtei":  mtei_char,
        "Mymr":  mymr_char,
        "Nkoo":  nkoo_char,
        "Olck":  olck_char,
        "Orya":  orya_char,
        "Phag":  phag_char,
        "Sinh":  sinh_char,
        "Sund":  sund_char,
        "Taml":  taml_char,
        "Telu":  telu_char,
        "Tfng":  tfng_char,
        "Thai":  thai_char,
        "Tibt":  tibt_char
    }

    hist_map = {}
    for i in range(0xFFFF):
        for key, prog in regex_map.items():
            if prog.match("{:04X}".format(i)):
                if key not in hist_map:
                    hist_map[key] = set()
                hist_map[key].add(chr(i))
    print("hist built.", file=sys.stderr)

    for line in sys.stdin:
        line = line.rstrip()
        line = re.sub(pattern, "", line)

        script_scores = []
        for key, st in hist_map.items():
            ln = len([c for c in line if c in st])
            script_scores.append((ln, key))

        predicted_script = sorted(script_scores, reverse=True)[0][1]
        if args.filter_mode:
            print(f"{predicted_script}\t{line}")
        else:
            print(predicted_script)


if __name__ == "__main__":
    main()
