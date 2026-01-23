"""
RecallVision - Wikipedia Data Collection Script
Fetches and saves Wikipedia articles with caching support
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import WikipediaFetcher
from src.utils import get_logger, safe_filename, file_exists, write_text_file, format_file_size
from src.config import config, RAW_DATA_DIR

logger = get_logger(__name__)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Search and scrape Wikipedia articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_wiki.py --query "Artificial Intelligence"
  python scripts/fetch_wiki.py --query "Machine Learning" --force
  python scripts/fetch_wiki.py --query "Python" --output custom_data/raw
        """
    )
    
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Topic to search on Wikipedia'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=str(RAW_DATA_DIR),
        help=f'Output directory for saved text file (default: {RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Output filename (default: auto-generated from query)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    args = parser.parse_args()
    
    if args.filename:
        filename = args.filename
    else:
        safe_name = safe_filename(args.query)
        filename = f"{safe_name}.txt"
    
    output_path = Path(args.output) / filename
    
    if file_exists(output_path) and not args.force:
        logger.info(f"File already exists: {output_path}")
        logger.info("Use --force to re-download")
        print(f"\n✅ Using cached file: {output_path}")
        print(f"   Use --force flag to re-download")
        return 0
    
    print("=" * 70)
    print("📚 WIKIPEDIA ARTICLE FETCHER")
    print("=" * 70)
    print(f"Query: {args.query}")
    if file_exists(output_path) and args.force:
        print("Mode: Force re-download")
    print("=" * 70)
    print()
    
    fetcher = WikipediaFetcher()
    
    logger.info(f"Fetching Wikipedia article for: {args.query}")
    url, text = fetcher.fetch(args.query)
    
    if not url:
        print("\n❌ Failed to find Wikipedia article")
        logger.error("Article search failed")
        return 1
    
    print(f"✅ Found article: {url}")
    
    if not text:
        print("\n❌ Failed to scrape article content")
        logger.error("Article scraping failed")
        return 1
    
    print(f"✅ Scraped {len(text):,} characters")
    
    logger.info(f"Saving to: {output_path}")
    success = write_text_file(output_path, text)
    
    if not success:
        print("\n❌ Failed to save file")
        logger.error("File save failed")
        return 1
    
    file_size = output_path.stat().st_size
    print(f"✅ Saved to: {output_path}")
    print(f"   File size: {format_file_size(file_size)}")
    
    print()
    print("=" * 70)
    print("✅ DATA COLLECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
