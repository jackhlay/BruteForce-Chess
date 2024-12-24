FROM golang:1.22.10-bullseye

WORKDIR /app

COPY go.mod go.sum ./
COPY *.go ./
COPY ./run.sh ./run.sh

RUN go mod tidy && go mod download
RUN go build -o Crawler

EXPOSE 3000

RUN chmod +x run.sh
ENTRYPOINT ["/run.sh"]