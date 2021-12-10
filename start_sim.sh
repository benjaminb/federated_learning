#!/bin/sh
python3 -c "import nltk; nltk.download('punkt')"

pip3 install -r requirements.txt

(trap 'kill 0' SIGINT; python3 simulate_server.py & python3 simulate_users.py)