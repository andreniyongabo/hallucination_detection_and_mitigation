import glob
import hashlib

DESTDIR = "/private/home/chau/wmt21/multilingual_bin/bilingual/en_is.wmt_mined.joined.32k"


WMT_ONLY = [
#    "en-ha",
#    "en-is",
    "en-ja",
    "en-ps",
    "en-km",
    "en-ta",
    "en-cs",
    "en-ru",
    "en-zh",
    "en-de",
    "en-pl",
]

WMT_NO_CJK = [
#    "en-ha",
#    "en-is",
    "en-ps",
    "en-km",
    "en-ta",
    "en-cs",
    "en-ru",
    "en-de",
    "en-pl",
]
test_set = set()
valid_set = set()

for direction in ['en-is']:
    src, tgt = direction.split('-')
    direction = direction.replace('-', '_')
    with open(f"{DESTDIR}/test.{direction}.{src}") as source_in_f, \
            open(f"{DESTDIR}/test.{direction}.{tgt}") as target_in_f  :
        for src_line, tgt_line in zip(source_in_f, target_in_f):
            concated = f"{src_line} {tgt_line}".lower()
            concated = ' '.join(concated.split(' ')[2:])
            hashed = hashlib.md5(concated.encode()).hexdigest()
            test_set.add(hashed)
    with open(f"{DESTDIR}/valid.{direction}.{src}") as source_in_f, \
            open(f"{DESTDIR}/valid.{direction}.{tgt}") as target_in_f  :
        for src_line, tgt_line in zip(source_in_f, target_in_f):
            concated = f"{src_line} {tgt_line}".lower()
            concated = ' '.join(concated.split(' ')[2:])
            hashed = hashlib.md5(concated.encode()).hexdigest()
            test_set.add(hashed)

train_set = set()
dups_count = 0
for direction in ['en-is']:
    src, tgt = direction.split('-')
    direction = direction.replace('-', '_')
    with open(f"{DESTDIR}/train.spm.clean.{direction}.{src}") as source_in_f, \
            open(f"{DESTDIR}/train.spm.clean.{direction}.{tgt}") as target_in_f:
        for src_line, tgt_line in zip(source_in_f, target_in_f):
            concated = f"{src_line} {tgt_line}".lower()
            concated = ' '.join(concated.split(' ')[2:])
            hashed = hashlib.md5(concated.encode()).hexdigest()
            if hashed in test_set:
                print(f"{direction} In TEST: {src_line}".strip())
                dups_count += 1
            if hashed in valid_set:
                print(f"{direction} In VALID: {src_line}".strip())
                dups_count += 1
            if hashed in train_set:
                print(f"{direction} In TRAIN: {src_line}".strip())
                dups_count += 1
            train_set.add(hashed)

print("Dups count:", dups_count)
print("Final count:", len(train_set))
