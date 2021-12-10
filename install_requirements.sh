#!/bin/sh
pip3 install -r requirements.txt

echo "[install_requirements.txt] Finished installing from requirements.txt"

python3 -c "import nltk; nltk.download('punkt')"
