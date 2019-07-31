FROM python:3.6
ENV PYTHONUNBUFFERED=1
COPY ./classification-appeals /srv/www/classification-appeals
WORKDIR /srv/www/classification-appeals
RUN pip install --upgrade -r requirements.txt
RUN python -c 'import nltk; nltk.download("punkt")'
