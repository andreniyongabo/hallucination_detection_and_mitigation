#!/usr/bin/env bash


if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. Provide the input and output path"
    exit
fi

ANALYZE_FOLDER=$1
OUTPUT_FOLDER=$2


pretty_folder_name() {
    echo $1 | sed 's/\/large_experiments\/nllb\/mmt\/data\/monolingual\///'
}

analyze_folder() {
    FOLDER=$1
    OUT_FOLDER=$2
    i=0
    # cat debugfiles.txt | while read f
    DRY_RUN=''

    find $FOLDER -type f | grep paracrawl9 | while read f
    do
        f_p=$(pretty_folder_name $f)
        echo $f_p

        # continue

        filename=$(basename -- "$f")
        dirname="$(dirname -- $f)"
        imm_dirname=$(echo $dirname | rev | cut -f 1 -d"/" | rev)
        extension="${filename##*.}"
        filename_2=$(basename -- "$f" ".$extension")
        lang_name=$(echo $imm_dirname | cut -f 1 -d"_")
        datasource_name=$(echo $filename_2 | cut -f 1 -d".")

        # if [ $imm_dirname == "raw" ]; then
        #     continue
        # fi

        if [ $lang_name == ".bash" ]; then
            continue
        fi

        CAT_COMMAND=""
        if [ $extension == "xz" ]; then
            CAT_COMMAND="xzcat"
        elif [ $extension == "txt" ]; then
            CAT_COMMAND="cat"
        elif [ $extension == "gz" ]; then
            CAT_COMMAND="zcat"
        fi

        # echo "datasource_name=$datasource_name"
        # echo "lang_name=$lang_name"

        # echo " "
        # continue

        count_out_filename="$OUT_FOLDER/$lang_name/${datasource_name}.${lang_name}.scripts_counts"
        out_dirname="$(dirname -- $count_out_filename)"
        mkdir -p $out_dirname

        if [ ! -z $CAT_COMMAND ]; then
            $CAT_COMMAND $f | parallel -j40 -k --pipe /private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier/script_detector.py 2> /dev/null  1> $count_out_filename
            count_results=$(cat $count_out_filename | sort | uniq -c | sort -n -r)

            NB_SCRIPT_RESULTS=2
            NB_SCRIPT_i=0
            echo "$count_results" | while read ctr_result
            do
                SCRIPT_NAME=$(echo $ctr_result | cut -f 2 -d" ")
                SCRIPT_SIZE=$(echo $ctr_result | cut -f 1 -d" ")

                out_filename="$OUT_FOLDER/$lang_name/${datasource_name}.${lang_name}_${SCRIPT_NAME}.$extension"

                file_ctr=0
                while [ -f "$out_filename" ]; do
                    # echo "$out_filename exists."
                    file_ctr=$((file_ctr+1))
                    out_filename="$OUT_FOLDER/$lang_name/${datasource_name}.${file_ctr}.${lang_name}_${SCRIPT_NAME}.$extension"
                done


                NB_SCRIPT_i=$((NB_SCRIPT_i+1))
                ignore_str=''
                if [ $NB_SCRIPT_i -gt $NB_SCRIPT_RESULTS ]; then
                    ignore_str="(ignored)"
                    # break
                fi
                # if [ -z $DRY_RUN ]; then
                #     ignore_str="(dryrun)"
                # fi

                if [ -z $ignore_str ]; then
                    # echo "processing"
                    paste $count_out_filename <($CAT_COMMAND $f) | grep "^${SCRIPT_NAME}" | cut -f 2- | xz > $out_filename
                fi

                out_filename_p=$(pretty_folder_name $out_filename)
                echo -e "\t=> $out_filename_p\t$SCRIPT_SIZE\t$ignore_str"

            done
            rm $count_out_filename
        else
            echo "Unknown extension"
        fi

        echo " "


        # i=$((i+1))
        # if [ $i -gt 5 ]; then
        #     break
        #     :
        # fi
    done
}


analyze_folder $ANALYZE_FOLDER $OUTPUT_FOLDER

