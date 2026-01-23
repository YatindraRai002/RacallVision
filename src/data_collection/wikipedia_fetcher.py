import requests
from bs4 import BeautifulSoup
from typing import Optional, Tuple
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.text_cleaning import clean_wikipedia_text
from src.config import config

logger = get_logger(__name__)


class WikipediaFetcher:
    """Fetches and scrapes Wikipedia articles"""
    
    def __init__(self, timeout=None):
        # use config timeout if not provided
        self.timeout = timeout if timeout else config.WIKI_REQUEST_TIMEOUT
        # need a proper user agent or wikipedia blocks us
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_article(self, query: str) -> Optional[str]:
        """Search for a Wikipedia article and return the URL"""
        logger.info(f"Searching for Wikipedia article: '{query}'")
        
        # try direct URL first - faster if it works
        formatted_query = query.replace(' ', '_')
        direct_url = f"https://en.wikipedia.org/wiki/{formatted_query}"
        
        try:
            response = requests.head(direct_url, headers=self.headers, 
                                   timeout=self.timeout, allow_redirects=True)
            if response.status_code == 200:
                logger.info(f"Found Wikipedia page (direct match): {direct_url}")
                return direct_url
        except requests.RequestException as e:
            logger.debug(f"Direct URL failed: {e}")
        
        # fallback to search API
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': 1,
                'namespace': 0,
                'format': 'json'
            }
            
            response = requests.get(search_url, params=params, 
                                  headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            results = response.json()
            
            # opensearch returns [query, [titles], [descriptions], [urls]]
            if len(results) > 3 and results[3]:
                wiki_url = results[3][0]
                logger.info(f"Found Wikipedia page (via search): {wiki_url}")
                return wiki_url
            else:
                logger.warning(f"No Wikipedia article found for '{query}'")
                return None
                
        except requests.Timeout:
            logger.error("Search request timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"Network error during search: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return None
    
    def scrape_article(self, url: str) -> Optional[str]:
        """Scrape the main text content from a Wikipedia article"""
        logger.info(f"Scraping article from: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # main content is in this div
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                logger.error("Could not find article content")
                return None
            
            paragraphs = content_div.find_all('p')
            
            # skip these sections - they're not useful for our purposes
            ignore_sections = ['references', 'external links', 'see also',
                             'notes', 'bibliography', 'further reading']
            
            text_content = []
            for para in paragraphs:
                # check if we're in a section we should skip
                parent_heading = para.find_previous(['h2', 'h3'])
                if parent_heading:
                    heading_text = parent_heading.get_text().lower().strip()
                    if any(ignore in heading_text for ignore in ignore_sections):
                        continue
                
                text = para.get_text().strip()
                # only keep paragraphs that are long enough
                if text and len(text) > config.WIKI_MIN_PARAGRAPH_LENGTH:
                    text_content.append(text)
            
            if not text_content:
                logger.error("No substantial content found in article")
                return None
            
            full_text = '\n\n'.join(text_content)
            cleaned_text = clean_wikipedia_text(full_text)
            
            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split())
            logger.info(f"Extracted {char_count:,} characters ({word_count:,} words)")
            
            return cleaned_text
            
        except requests.Timeout:
            logger.error("Request timed out while scraping article")
            return None
        except requests.RequestException as e:
            logger.error(f"Network error while scraping: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while scraping: {e}")
            return None
    
    def fetch(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Search and scrape a Wikipedia article in one go"""
        # first find the article
        url = self.search_article(query)
        if not url:
            return None, None
        
        # then scrape it
        text = self.scrape_article(url)
        if not text:
            return url, None
        
        return url, text
