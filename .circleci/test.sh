#!/usr/bin/env bash

set -e
set -u
set -o pipefail

sudo apt-get install libopenblas-dev

source "$(curl -fsS  --retry 3 https://dlang.org/install.sh | bash -s $1 --activate)"
dub test --build=unittest-cov

if [ "$DC" = dmd ]; then
    make doc;
    mv generated-docs docs;
    bash <(curl -s https://codecov.io/bash) -s "source-grain-*.lst";
fi

cat dub.selections.json
