#!/bin/bash

# taken from respondo

DATAFILES=(
	data_0.1.0.tar.gz
)
#
# -----
#

THISDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$THISDIR"
echo "Updating testdata ... please wait."

if which sha256sum &> /dev/null; then
	sha256sum --ignore-missing -c SHA256SUMS || exit 1
fi

for file in ${DATAFILES[@]}; do
    tar -xzf $file
done

exit 0
