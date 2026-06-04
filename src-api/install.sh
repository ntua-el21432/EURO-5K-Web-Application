#!/bin/bash
set -e 

declare -A langMappings=()
while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    langMappings[$key]=$value
done < <(jq -j 'to_entries[] | (.key, "\u0000", .value, "\u0000")' < mappings.json)
availableLangs=${!langMappings[@]}

# echo "Insert the list of models to use, separated by space (eg. 'bert_fft llama')."
# echo "Available models: ${availableLangs[@]}."
# echo "Leave blank to download all the ${#langMappings[@]} languages (50GB of data approximately)."
# read -a selectedLangs
selectedLangs=( "$@" )

declare -a okLangs=()
if [ ! ${#selectedLangs[@]} -eq 0 ]; then
    for l in ${selectedLangs[@]}; do
        if [[ ! ${availableLangs[@]} =~ $l ]]; then
            echo "Lang $l does not exist, ignoring"
            continue
        fi
        okLangs+=($l)
    done
else
    okLangs=${availableLangs[@]}
fi

if [ ${#okLangs[@]} -eq 0 ]; then
    echo "No languages selected, exiting"
    exit 1
fi

mkdir -p models
cd models
echo "Installing languages: ${okLangs[@]}"
for l in ${okLangs[@]}; do
    modelName=${langMappings[$l]}
    echo "Installing model for ${langMappings[$l]^}"
    curl -L -f -H --output "${modelName}.tar.gz" "https://huggingface.co/bill-kotronis/reporting_obligations_thesis_models/resolve/main/${modelName}.tar.gz?download=true"
    mkdir -p "${modelName}"
    tar xzf "${modelName}.tar.gz" -C "${modelName}" --strip-components=2
    rm "${modelName}.tar.gz"
done
cd ..
