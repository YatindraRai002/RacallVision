

import requests
import os
from typing import Optional


def translate_text(text: str, source_language: str = "auto", target_language: str = "en-IN", api_key: Optional[str] = None) -> str:
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv("SARVAM_API_KEY")
    
    if not api_key:
        raise ValueError("Sarvam API key required. Set SARVAM_API_KEY env variable.")
    
    url = "https://api.sarvam.ai/translate"
    
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    payload = {
        "input": text,
        "source_language_code": source_language,
        "target_language_code": target_language,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True
    }
    
    try:
        print(f"Translating text from {source_language} to {target_language}...")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        translated_text = result.get("translated_text", "")
        
        if not translated_text:
            return text
        
        return translated_text
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        raise
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise
    
    except Exception as e:
        print(f"Unexpected error during translation: {e}")
        raise


def main():
    """Test the translation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate text using Sarvam AI')
    parser.add_argument('text', type=str, help='Text to translate')
    parser.add_argument(
        '--source',
        type=str,
        default='auto',
        help='Source language code (default: auto)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='en-IN',
        help='Target language code (default: en-IN)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Sarvam API key (optional, uses SARVAM_API_KEY env var if not provided)'
    )
    
    args = parser.parse_args()
    
    try:
        translated = translate_text(
            text=args.text,
            source_language=args.source,
            target_language=args.target,
            api_key=args.api_key
        )
        
        print("\n" + "=" * 60)
        print("TRANSLATION RESULT")
        print("=" * 60)
        print(f"Original: {args.text}")
        print(f"Translated: {translated}")
    
    except Exception as e:
        print(f"\nTranslation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
