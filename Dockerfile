FROM python:3.8

ADD requirements.txt /app/requirements.txt

RUN python3 -m pip install -r /app/requirements.txt
RUN python3 -c "import nltk; nltk.download('punkt')"
COPY . /app

WORKDIR /app