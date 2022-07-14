# NLLB Languages Module

This module is a toolbox to list and merge tabular language information related to **No Language Left Behind** project.

[Learn more](<https://fb.workplace.com/groups/nllb.xfn/posts/308289640812765>)



## Installation

```
pip install .
```

## Usage

There are two main commands: `list` and `merge`. The `list` command prints static information about NLLB Languages, and `merge` command is a helper to merge two tabular data that can be in various forms (tsv, excel, etc.). Each one of these operations can be exported to various formats.

### `list` command

Without arguments, the program prints the list of NLLB-200 languages:
```bash
> nllblangs
ace_Arab    Acehnese (Arabic script)
ace_Latn    Acehnese (Latin script)
afr Afrikaans
aka Akan
sqi Albanian
amh Amharic
...
```


```bash
> nllblangs
```

is equivalent to:
```bash
> nllblangs list --columns "Code" "Language Name" --export simple
```

With the `--export` argument, you can export the selected list to stdout in a simple tsv format or python dictionary. You can also export to a file if you set a filename (.tsv or .xlsx):

```bash
> nllblangs --export simple       # prints a tsv in stdout
> nllblangs --export python       # prints a python dictionary in stdout
> nllblangs --export langs.tsv    # writes a tsv file
> nllblangs --export langs.xlsx   # writes an excel file
```

In order to print all possible column names, enter an invalid column name:

```bash
> nllblangs --columns "Column name unlikely to be there"

Column(s) not found: ['Column name unlikely to be there']

Use one of these columns:
    Language Name
    Current Data
    BCP47
    Code
    Script
    # Speakers
...
```


### `merge` command

This command helps to merge to tabular data into one. Suppose you have an excel file like this:

![Excel file with language data](/examples/nllb/nllblangs/docs/images/left.xlsx.png)


and a .tsv file like this:

right.tsv:
```tsv
pcm Nigerian Pidgin
oss Ossetic
roh Romansh
sme Northern Sami
skr Saraiki
```

Let's try to merge those files:

```bash
> nllblangs left.xlsx right.tsv
Available columns in left:
    0
    1
Available columns in right:
    0
    1
```

The program asks to select the key column on each file. In this case, we can simple select column `0`:
```bash
> nllblangs merge left.xlsx right.tsv --left-column 0 --right-column 0
sme Northern Sami   Northern Sami
skr Saraiki Saraiki
```

By default, the merge algorithm is set on `inner`. `--how` option allows to change this.

To keep rows in the left when the key is missing in the right:
```bash
> nllblangs merge left.xlsx right.tsv --left-column 0 --right-column 0 --how left
guz Gusii
haw Hawaiian
kln Kalenjin
khq Koyra Chiini Songhay
sme Northern Sami   Northern Sami
skr Saraiki Saraiki
syr Syriac
```

To keep both:
```bash
> nllblangs merge left.xlsx right.tsv --left-column 0 --right-column 0 --how outer
guz Gusii
haw Hawaiian
kln Kalenjin
khq Koyra Chiini Songhay
sme Northern Sami   Northern Sami
skr Saraiki Saraiki
syr Syriac
pcm     Nigerian Pidgin
oss     Ossetic
roh     Romansh
```

`--how` can be set to `left`, `right`, `outer`, `inner`, `cross`. For more information, look at [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html).

Like in the `list` command, you can export the result to a file with the `--export` command.

```bash
> nllblangs merge left.xlsx right.tsv --left-column 0 --right-column 0 --how outer --export merged.xlsx
Exported to merged.xlsx
```

#### Missing key warnings

Some warnings were not reported in this document for readability, but the program prints a few warnings/info in stderr:
```bash
> nllblangs merge left.xlsx right.tsv --left-column 0 --right-column 0 --how outer --export merged.xlsx
Warning: these values appear only on the left: {'haw', 'guz', 'khq', 'kln', 'syr'}
Warning: these values appear only on the right: {'oss', 'pcm', 'roh'}
Left size: 7
Right size: 5
Exported to merged.xlsx
```


You can use `--left-header` and `--right-header` arguments to offset the header for each file.
