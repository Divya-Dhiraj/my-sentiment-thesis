# web_scraper/Dockerfile
FROM python:3.11-slim

WORKDIR /scraper

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scraper.py .

CMD ["uvicorn", "scraper:app", "--host", "0.0.0.0", "--port", "8003"]
