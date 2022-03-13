FROM python:3.9-slim-buster

COPY dataset/behaviors.tsv apk/dataset/behaviors.tsv
COPY dataset/news.tsv apk/dataset/news.tsv
COPY . apk/

WORKDIR apk/

RUN apt-get update \
    && get upgrade -y \
    && get install -y git \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"] \
    && ["mlflow", "ui", "-h", "0.0.0.0", "-p", "5000"] 




    
