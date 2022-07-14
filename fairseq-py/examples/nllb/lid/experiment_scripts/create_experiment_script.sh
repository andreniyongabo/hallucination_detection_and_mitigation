#!/usr/bin/env bash


if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Provide the name of the experiment"
    exit
fi


if [[ $(hostname) =~ "devfair" ]]; then
    echo "On devfair"
    DATE_SIGNATURE=$(date -d "+9 hours" +"%Y-%m-%d-%H-%M")
else
    echo "On local"
    DATE_SIGNATURE=$(date +"%Y-%m-%d-%H-%M")
fi

echo "DATE_SIGNATURE=$DATE_SIGNATURE"
ORIGINAL_EXPERIMENT_NAME=$1

EXPERIMENT_NAME="$DATE_SIGNATURE-$ORIGINAL_EXPERIMENT_NAME"

SCRIPT=`realpath $0`
HERE=`dirname $SCRIPT`

NEW_SCRIPT_FILE="$HERE/$EXPERIMENT_NAME.sh"

touch $NEW_SCRIPT_FILE
chmod +x $NEW_SCRIPT_FILE

cat > ${NEW_SCRIPT_FILE} <<- EOM
#!/bin/bash


EXPERIMENT_NAME="${EXPERIMENT_NAME}"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/\$EXPERIMENT_NAME"

mkdir -p \$EXPERIMENT_FOLDER

EOM

