FROM golang:1.22.10-bullseye

WORKDIR /app

COPY go.mod go.sum ./

RUN go get

RUN go mod tidy && go mod download

COPY *.go ./

RUN go build -o Crawler

EXPOSE 3000

CMD ["./Crawler"]