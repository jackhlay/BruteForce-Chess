FROM python:3.13-alpine

# Use apk instead of apt for Alpine
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers

WORKDIR /app

COPY reqs.txt .

# Install dependencies before copying all code to use cache
RUN pip install --no-cache-dir -r reqs.txt

COPY engin.py bot.py ./

# useradd is not in Alpine; use adduser
RUN adduser -D bot
USER bot

ENV LIP=""
ENV TIME_LIMIT="3.247"
ENV MAX_DEPTH="64"

CMD ["python", "-u", "bot.py"]
