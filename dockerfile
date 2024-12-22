FROM golang:1.22.10-bullseye

WORKDIR /app

COPY go.mod go.sum ./
COPY *.go ./

RUN go mod tidy && go mod download
RUN go build -o Crawler

EXPOSE 3000

# CMD ["./Crawler", "move=17"] # Getting rid of this to use dynamic container arguments in cluster