FROM python:3.13-alpine

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY reqs.txt .

RUN pip install --no-cache-dir -r reqs.txt

COPY engin.py bot.py ./

RUN useradd -m bot
USER bot

ENV LICHESS_TOKEN=""
ENV CONCURRENCY="3"
ENV TIME_LIMIT="3.247"


CMD ["python", "-u", "bot.py"]