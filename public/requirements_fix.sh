#!/bin/bash
# this script takes a version unspecified requirements file and creates a new requirements file with only the specified versions of the packages

REQ_FILE="public/requirements_seed.txt"
TEMP_FILE="temp_requirements.txt"

pip install -r $REQ_FILE --quiet
pip freeze >$TEMP_FILE

>requirements.txt

while IFS= read -r package || [[ -n "$package" ]]; do
    # Extract the package name, stripping any version or comparison operators
    clean_package_name=$(echo $package | sed 's/>=.*//; s/<=.*//; s/==.*//; s/>.*//; s/<.*//')
    # Search for the clean package name in the full list, ignoring case, and add it to the new requirements file
    grep -i "^$clean_package_name==" $TEMP_FILE >>requirements.txt
done < <(
    cat $REQ_FILE
    echo ""
)

rm $TEMP_FILE

cat requirements.txt
