import re
import csv
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    pipeline
)

# === Model Initialization === #
_SUMMARIZER_MODEL = "human-centered-summarization/financial-summarization-pegasus"
_tokenizer = PegasusTokenizer.from_pretrained(_SUMMARIZER_MODEL)
_summarizer_model = PegasusForConditionalGeneration.from_pretrained(_SUMMARIZER_MODEL)
_sentiment_model = pipeline("sentiment-analysis")


def find_news_links(keyword: str) -> List[str]:
    """Searches Google News for links related to a keyword."""
    query_url = f"https://www.google.com/search?q=yahoo+finance+{keyword}&tbm=nws"
    response = requests.get(query_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
    return links
