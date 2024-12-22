FROM golang:1.22.10-bullseye

WORKDIR /app

COPY go.mod go.sum ./
COPY *.go ./
COPY run.sh ./run.sh

RUN go mod tidy && go mod download
RUN go build -o Crawler

# RUN chmod +x /entrypoint.sh
# ENTRYPOINT ["/run.sh"]

EXPOSE 3000

CMD ["./Crawler", "move=17"]