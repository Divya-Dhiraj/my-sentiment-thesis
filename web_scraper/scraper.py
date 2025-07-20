# web_scraper/scraper.py

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from typing import Optional # NEW IMPORT

app = FastAPI()

# UPDATE scrape endpoint to accept optional selector
@app.get("/scrape")
def scrape_title_or_data(url: str, css_selector: Optional[str] = None):
    """
    A simple endpoint that scrapes the title of a given URL,
    or specific content using a CSS selector if provided.
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is missing.")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # If a selector is provided, try to find and return that content
        if css_selector:
            found_element = soup.select_one(css_selector)
            if found_element:
                # You might need to refine this based on what exactly you want (text, inner HTML, attribute)
                extracted_content = found_element.get_text(strip=True)
                return {"url": url, "extracted_content": extracted_content}
            else:
                return {"url": url, "extracted_content": f"No element found for selector: {css_selector}"}
        else:
            # Otherwise, just return the title as before
            title = soup.title.string if soup.title else "No title found"
            return {"url": url, "title": title.strip()}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error or invalid URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during scraping: {e}")