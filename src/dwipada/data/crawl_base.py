"""Base crawler with shared utilities for all Telugu poetry web crawlers.

Extracts common patterns: HTTP fetching with retry, HTML cleaning,
filename sanitization, and text cleaning.
"""

import re
import time
import warnings
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from dwipada.paths import DATA_DIR

# Shared request configuration
TIMEOUT = 60
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def suppress_ssl_warnings():
    """Suppress SSL certificate verification warnings."""
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def fetch_page(url: str, retries: int = 3, verify_ssl: bool = False,
               timeout: int = TIMEOUT) -> Optional[str]:
    """Fetch HTML content from a URL with configurable retry and SSL settings.

    Args:
        url: The URL to fetch.
        retries: Number of retry attempts with exponential backoff.
        verify_ssl: Whether to verify SSL certificates.
        timeout: Request timeout in seconds.

    Returns:
        HTML content as string, or None on failure.
    """
    for attempt in range(retries):
        try:
            response = requests.get(
                url, headers=HEADERS, timeout=timeout, verify=verify_ssl
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    print(f"  ERROR: Failed to fetch {url} after {retries} attempts")
    return None


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Sanitize a string for use as a filename.

    Removes invalid characters and limits length.
    """
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '', name)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized[:max_length].strip() if len(sanitized) > max_length else sanitized.strip()


def clean_html_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove common noise elements from parsed HTML.

    Removes: page numbers, footnote superscripts, ws-noexport divs,
    edit section links. Converts <br> tags to newlines.
    """
    for pagenum in soup.find_all('span', class_='pagenum'):
        pagenum.decompose()
    for sup in soup.find_all('sup', class_='reference'):
        sup.decompose()
    for noexport in soup.find_all(class_='ws-noexport'):
        noexport.decompose()
    for edit_link in soup.find_all('span', class_='mw-editsection'):
        edit_link.decompose()
    for br in soup.find_all('br'):
        br.replace_with('\n')
    return soup


def clean_text(text: str) -> str:
    """Clean extracted text by removing citations, page numbers, and excess whitespace."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


def find_content_div(soup: BeautifulSoup):
    """Find the main content div (Wikisource or MediaWiki format)."""
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        content_div = soup.find('div', class_='prp-pages-output')
    return content_div


def extract_section_content(section_soup: BeautifulSoup) -> str:
    """Extract clean text from a section's HTML (poem divs + p tags)."""
    soup = BeautifulSoup(str(section_soup), 'lxml')
    clean_html_content(soup)

    all_text_parts = []
    poems = soup.find_all('div', class_='poem')
    for poem in poems:
        poem_text = poem.get_text()
        if poem_text.strip():
            all_text_parts.append(poem_text)

    for p in soup.find_all('p'):
        if p.find_parent('div', class_='poem'):
            continue
        p_text = p.get_text()
        if p_text.strip():
            all_text_parts.append(p_text)

    if all_text_parts:
        text = '\n'.join(all_text_parts)
    else:
        text = soup.get_text()

    return clean_text(text)
