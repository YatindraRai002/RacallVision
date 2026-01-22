# Task 1: Wikipedia scraper

import argparse
import requests
from bs4 import BeautifulSoup
import os
import urllib.parse


def search_wikipedia(query):
    # Try to find Wikipedia article URL
    # Method 1: Try direct URL construction
    formatted_query = query.replace(' ', '_')
    direct_url = f"https://en.wikipedia.org/wiki/{formatted_query}"
    
    try:
        response = requests.head(direct_url, timeout=5)
        if response.status_code == 200:
            print(f"Found Wikipedia article: {direct_url}")
            return direct_url
    except:
        pass
    
    # Method 2: Use Wikipedia search API
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'query' in data and 'search' in data['query'] and len(data['query']['search']) > 0:
            title = data['query']['search'][0]['title']
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            print(f"Found Wikipedia article: {url}")
            return url
    
    except Exception as e:
        print(f"Error during search: {e}")
    
    print("No Wikipedia article found for the query.")
    return None


def scrape_wikipedia_article(url):
    # Scrape article text
    try:
        # Send GET request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content div
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if not content_div:
            print("Could not find main content div")
            return ""
        
        # Extract all paragraphs
        paragraphs = content_div.find_all('p')
        
        # Combine paragraph text
        text_content = []
        for para in paragraphs:
            text = para.get_text().strip()
            if text and len(text) > 20:  # Only add substantial paragraphs
                text_content.append(text)
        
        full_text = '\n\n'.join(text_content)
        print(f"Scraped {len(full_text)} characters from the article")
        
        return full_text
    
    except Exception as e:
        print(f"Error during scraping: {e}")
        return ""


def save_to_file(text, filename='data/scraped_text.txt'):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Text saved to {filename}")
    
    except Exception as e:
        print(f"Error saving file: {e}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Search and scrape Wikipedia articles for a given topic'
    )
    parser.add_argument(
        'query',
        type=str,
        help='The topic/query to search for on Wikipedia'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/scraped_text.txt',
        help='Output file path (default: data/scraped_text.txt)'
    )
    
    args = parser.parse_args()
    
    print(f"Searching for Wikipedia article on: {args.query}")
    
    # Step 1: Search for Wikipedia article
    wiki_url = search_wikipedia(args.query)
    
    if not wiki_url:
        print("Failed to find a Wikipedia article. Exiting.")
        return
    
    # Step 2: Scrape the article
    text_content = scrape_wikipedia_article(wiki_url)
    
    if not text_content:
        print("Failed to scrape content. Exiting.")
        return
    
    # Step 3: Save to file
    save_to_file(text_content, args.output)
    
    print("\nData collection completed successfully!")


if __name__ == "__main__":
    main()
