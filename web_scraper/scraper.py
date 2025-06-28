# web_scraper/scraper.py
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI

app = FastAPI()

@app.get("/scrape")
def scrape_title(url: str):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # This is a simple example that just gets the title of the page
        return {"url": url, "title": soup.title.string}
    except Exception as e:
        return {"error": str(e)}