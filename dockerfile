FROM golang:1.22.10-bullseye

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod tidy && go mod download

COPY . .

RUN go build -o Crawler

EXPOSE 3000

CMD ["./Crawler"]