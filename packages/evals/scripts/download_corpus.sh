#!/bin/bash
# Downloads FastAPI documentation corpus for eval benchmarks
# Run from packages/evals/

set -e

echo "Downloading FastAPI docs corpus..."
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/fastapi/fastapi.git /tmp/fastapi-source

cd /tmp/fastapi-source
git sparse-checkout set docs/en/docs

cp -r docs/en/docs/* $OLDPWD/corpus/
cd $OLDPWD
rm -rf /tmp/fastapi-source

echo "Done. Corpus available at packages/evals/corpus/"
