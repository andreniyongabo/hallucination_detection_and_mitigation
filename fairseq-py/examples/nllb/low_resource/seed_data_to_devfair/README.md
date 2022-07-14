
# Seed Data Utils

The scripts in this folder combine translated `.tsv` files from vendors and combine them per language to create bitext and monolingual files with NLLB folder/file naming conventions.



## Step 0 - Gather all files from vendors

Put every `.tsv` file from vendors under a folder (the script browses the folder recursively so you can create folders as you want as long as all files are here).

For example:


```
- path/to/Delivery
    - Delivery Moravia
        - nov 17
            - mri_sentences_6193.tsv
        - nov 19
            - ace-Latn_sentences_2000.tsv
            - bjn-Latn_sentences_2000.tsv
    - Delivery LionBridge
        - 2021-10-21
            - mag_sentences_194.tsv
        ...
```

The folder names are ignored.

The tsv files look like this:

```
target_lang	source_id	context_url	sentence	translation	translator_id
crh_Latn	000409276	https://en.wikipedia.org/wiki/Quadratic_equation	His solution was largely based on Al-Khwarizmi's work.	Onıñ çezilişi, çoqusında El-Harezminiñ işine esaslanğan.	81916
```

the filenames are like this: `ace_Latn_sentences_2000.tsv`



## Step 1 - Concatanate per lang

This script takes a folder like this:
```
- path/to/Delivery
    - Delivery Moravia
        - nov 17
            - mri_sentences_6193.tsv
        - nov 19
            - ace-Latn_sentences_2000.tsv
            - bjn-Latn_sentences_2000.tsv
        - nov-25
            - ace-Latn_sentences_1000.tsv
    - Delivery LionBridge
        - 2021-10-21
            - mag_sentences_194.tsv
        ...
```

and creates a folder like this:

```
- per-lang
    - ace_Latn.tsv
    - bjn_Latn.tsv
    - mag.tsv
    - mri.tsv
    ...
```

Run this:
```
01-extract-seed-data.py --vendors-folder path/to/Delivery --output per-lang
```



## Step 2 - Create NLLB-formatted folders and files

This script extracts translations from tsv files, compresses them and puts them in the NLLB-compliant form.

It takes a folder like this (create in the previous step):
```
- per-lang
    - ace_Latn.tsv
    - bjn_Latn.tsv
    - mag.tsv
    - mri.tsv
    ...
```

and a prefix name (for example `fbseed20211130`), then creates NLLB-compliant folder/files:

```
- bitext-folder
    - ace_Latn-eng
        - fbseed20211130.ace_Latn.gz
        - fbseed20211130.eng.gz
    - bam-eng
        - fbseed20211130.bam.gz
        - fbseed20211130.eng.gz
    ...
- monolingual-folder
    - ace_Latn
        - fbseed20211130.ace_Latn.xz
    - bam
        - fbseed20211130.bam.xz
    ...
```




Run this:

```
02-extract-seed-data-to-nllb-format.sh <per-lang-folder-created-in-step-1> <prefix> <bitext-folder-output> <monolingual-folder-output>
```

For example:
```
02-extract-seed-data-to-nllb-format.sh per-lang fbseed20211130 bitext-folder monolingual-folder
```


## Step 3 - Send to devfair


```
rsync -avr bitext-folder devfair0111:/path/to/bitext-folder/
rsync -avr monolingual-folder devfair0111:/path/to/monolingual-folder/
```
