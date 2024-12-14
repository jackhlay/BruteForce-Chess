FROM golang:1.22.0-alpine3.13

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod tidy && go mod download

COPY . .

RUN go build -o Crawler

EXPOSE 3000

cmd ["./Crawler -move= 17"]