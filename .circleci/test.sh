#!/usr/bin/env bash

set -e
set -u
set -o pipefail

sudo apt-get install libopenblas-dev libzmq5-dev cmake

source "$(curl -fsS  --retry 3 https://dlang.org/install.sh | bash -s $1 --activate)"
dub test --build=unittest-cov
dub build --build=release --config=example-mnist
dub build --build=release --config=example-mnist-cnn
dub build --build=release --config=example-cifar
dub build --build=release --config=example-char-rnn
dub build --build=release --config=example-ptb

# ldc2 causes linker error with drepl https://github.com/dlang-community/drepl/issues/39
if [ "$DC" = dmd ]; then
    dub build --build=release --config=repl;
    dub build --build=release --config=jupyterd;

    make doc;
    mv generated-docs docs;
    bash <(curl -s https://codecov.io/bash) -s "source-grain-*.lst";
fi

else
    echo "skipping make repl jupyterd in ${DC}";
fi

cat dub.selections.json
