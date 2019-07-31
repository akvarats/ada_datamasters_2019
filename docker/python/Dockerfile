FROM python:3.6
ENV PYTHONUNBUFFERED=1
COPY ./classification-appeals /srv/www/classification-appeals
WORKDIR /srv/www/classification-appeals
RUN pip install --upgrade -r requirements.txt
RUN python -c 'import nltk; nltk.download("punkt")'
RUN mkdir -p models/180; wget http://vectors.nlpl.eu/repository/11/180.zip -O tmp.zip; unzip -o -d models/180 tmp.zip; rm tmp.zip
