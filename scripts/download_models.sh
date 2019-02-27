#!/usr/bin/env bash

say() {
 echo "$@" | sed \
         -e "s/\(\(@\(red\|green\|yellow\|blue\|magenta\|cyan\|white\|reset\|b\|u\)\)\+\)[[]\{2\}\(.*\)[]]\{2\}/\1\4@reset/g" \
         -e "s/@red/$(tput setaf 1)/g" \
         -e "s/@green/$(tput setaf 2)/g" \
         -e "s/@yellow/$(tput setaf 3)/g" \
         -e "s/@blue/$(tput setaf 4)/g" \
         -e "s/@magenta/$(tput setaf 5)/g" \
         -e "s/@cyan/$(tput setaf 6)/g" \
         -e "s/@white/$(tput setaf 7)/g" \
         -e "s/@reset/$(tput sgr0)/g" \
         -e "s/@b/$(tput bold)/g" \
         -e "s/@u/$(tput sgr 0 1)/g"
}

# check current directory
current_dir=${PWD##*/}
if [ "$current_dir" == "scripts" ]; then
    say @red[["This scripts should be executed from the root folder as: ./scripts/download_models.sh"]]
    exit
fi

CHECKPOINTS_DIR="checkpoints"
MODEL_PATH="${CHECKPOINTS_DIR}/east_icdar2015_resnet_v1_50_rbox"


if [ ! -d ${MODEL_PATH} ] ; then

    if [ ! -d ${CHECKPOINTS_DIR} ] ; then
        echo "Creating checkpoints directory"
        mkdir ${CHECKPOINTS_DIR}
    fi

    echo "Downloading pre-trained model file"
    python ./scripts/download_gdrive.py 11n1Ccs9-XRaijdsgkvVbUXnZnAbUXWd0 "${MODEL_PATH}.zip"

    echo "Unpacking..."
    unzip "${MODEL_PATH}.zip" -d ${CHECKPOINTS_DIR}

    echo "Removing intermediate files..."
    rm "${MODEL_PATH}.zip"

    echo "Done!"

else
    echo "model checkpoint already present"
fi

