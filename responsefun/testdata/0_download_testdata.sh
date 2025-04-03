#!/bin/bash

# taken from respondo

# SOURCE="https://wwwagdreuw.iwr.uni-heidelberg.de/responsefun_test_data/0.1.0"
DATAFILES=(
	data_0.1.0.tar.gz
)
#
# -----
#

THISDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$THISDIR"
echo "Updating testdata ... please wait."

# ToDo: temporary fix store testdata in repository.
# As soon as new server is online move testdata there
# download() {
# 	if which wget &> /dev/null; then
# 	    echo $@
# 		wget -w 1 -qN --show-progress --no-check-certificate $@
# 	else
# 		echo "wget not installed" >&2
# 		exit 1
# 	fi
# }

# download $(for file in ${DATAFILES[@]}; do echo $SOURCE/$file; done)

if which sha256sum &> /dev/null; then
	sha256sum --ignore-missing -c SHA256SUMS || exit 1
fi

for file in ${DATAFILES[@]}; do
    tar -xzf $file
done

exit 0