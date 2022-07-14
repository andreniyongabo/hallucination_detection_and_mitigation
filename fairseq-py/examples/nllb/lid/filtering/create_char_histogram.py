#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import string

# from /private/home/edunov/t2t_filter/


fn=sys.argv[1]
of=sys.argv[2]
hist={}
with(open(fn, 'r', encoding='utf8')) as f:
    for line in f:
        for c in list(line.strip()):
            if c in hist:
                hist[c]=hist[c]+1
            else:
                hist[c]=1

nonprintable = set([chr(i) for i in range(128)]).difference(string.printable)
with open(of, encoding='utf-8', mode='w') as f:
    for k, v in sorted(hist.items(), key=lambda item: -item[1]):
        if k not in nonprintable:
            f.write(u'{} {}\n'.format(k, v))




