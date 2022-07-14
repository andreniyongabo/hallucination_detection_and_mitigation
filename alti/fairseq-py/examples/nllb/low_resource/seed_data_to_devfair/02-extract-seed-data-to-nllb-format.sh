#!/usr/bin/env bash


if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters. Usage: $0 <per-lang-folder> <prefix> <bitext-output-folder> <monolingual-output-folder>"
    exit
fi


ALL_FILES_FOLDER=$1
DATA_SOURCE_NAME=$2
BITEXT_FILES_FOLDER=$3
MONOLINGUAL_FILES_FOLDER=$4


ls -1 $ALL_FILES_FOLDER | while read f
do
    lang_name=$(echo $f | cut -f1 -d".");
    echo $lang_name

    bitext_folder_name=$( (
        echo "eng";
        echo "${lang_name}"
        ) | sort)
    bitext_folder_name=$(echo $bitext_folder_name | tr ' ' '-')

    bitext_output_file_folder="$BITEXT_FILES_FOLDER/$bitext_folder_name"
    mono_output_file_folder="$MONOLINGUAL_FILES_FOLDER/$lang_name"

    bitext_output_file_eng="$bitext_output_file_folder/$DATA_SOURCE_NAME.eng"
    bitext_output_file_xx="$bitext_output_file_folder/$DATA_SOURCE_NAME.$lang_name"
    mono_output_file_xx="$mono_output_file_folder/$DATA_SOURCE_NAME.$lang_name.xz"

    mkdir -p $bitext_output_file_folder
    mkdir -p $mono_output_file_folder
    TMP_FILE="tmp.tsv"
    cat "$ALL_FILES_FOLDER/$f" | grep -v -E "target_lang\tsource_id" | sort -u > $TMP_FILE
    cat $TMP_FILE | cut -f 4 > $bitext_output_file_eng
    cat $TMP_FILE | cut -f 5 > $bitext_output_file_xx
    cat $TMP_FILE | cut -f 5 | xz > $mono_output_file_xx # monolingual data is compressed with xz
    rm $TMP_FILE

    gzip $bitext_output_file_eng
    gzip $bitext_output_file_xx
done
