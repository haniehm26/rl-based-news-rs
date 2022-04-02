FROM python:3.9-slim-buster

COPY dataset/behaviors.tsv apk/dataset/behaviors.tsv
COPY dataset/news.tsv apk/dataset/news.tsv
COPY . apk/

WORKDIR apk/

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["uvicorn", "run:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"] \
    && ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"] 




    
