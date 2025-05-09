FROM python:3.12.8-slim-bullseye

WORKDIR /app

COPY model.py reqs.txt ./

RUN pip install --no-cache-dir -r reqs.txt

EXPOSE 8000

CMD ["python", "model.py"]