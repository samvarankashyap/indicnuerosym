#!/usr/bin/env python3
"""
Crawler for పల్నాటివీరచరిత్ర (Palanati Veera Charitra) from sahityasourabham.blogspot.com

Downloads all sections from 33 blog posts and saves them as organized .txt files.

Key features:
- 33 blog posts with search query URLs
- Content extracted from post-body div
- Section headings identified by bold text or h2/h3 tags
- Dwipada verse format preserved

Usage:
    python palanati_veera_charitra.py
"""

import re
import time
from bs4 import BeautifulSoup, NavigableString
from typing import List, Tuple, Optional
from urllib.parse import quote

from crawl_base import (
    DATA_DIR,
    suppress_ssl_warnings,
    fetch_page,
    sanitize_filename,
    clean_html_content,
    clean_text,
    find_content_div,
)

# Suppress SSL warnings
suppress_ssl_warnings()

# Configuration
BASE_URL = "https://sahityasourabham.blogspot.com/search?q="
SEARCH_QUERY_TEMPLATE = 'శ్రీనాధభట్టకృత " పల్నాటివీరచరిత్ర " -- ద్విపదకావ్యం - {}'
TOTAL_PAGES = 33

OUTPUT_DIR = DATA_DIR / "palanati_veera_charitra"


def get_url_for_page(page_num: int) -> str:
    """Generate the URL for a specific page number."""
    query = SEARCH_QUERY_TEMPLATE.format(page_num)
    encoded_query = quote(query)
    return BASE_URL + encoded_query


def extract_sections_from_post(post_body) -> List[Tuple[str, str]]:
    """Extract sections from a blog post body.

    The blog uses:
    - Centered red/large text for section headings
    - Blue text for prose explanations
    - Maroon/purple text for verse content
    - Each verse line in a separate div

    Returns:
        List of (title, content) tuples
    """
    sections = []
    content_html = str(post_body)

    # Find section headings - centered divs with large/red text
    # Pattern: <div style="text-align: center;"><span ... style="color: red; font-size: x-large;">HEADING</span></div>
    heading_pattern = re.compile(
        r'<div[^>]*text-align:\s*center[^>]*>.*?<span[^>]*>([^<]+)</span>.*?</div>',
        re.IGNORECASE | re.DOTALL
    )

    headings = []
    for match in heading_pattern.finditer(content_html):
        heading_text = match.group(1).strip()
        # Filter: must have Telugu characters and reasonable length
        if len(heading_text) >= 2 and any('\u0C00' <= c <= '\u0C7F' for c in heading_text):
            headings.append((heading_text, match.start(), match.end()))

    # Also look for h2/h3 headings
    for header_match in re.finditer(r'<h[23][^>]*>([^<]+)</h[23]>', content_html, re.IGNORECASE):
        heading_text = header_match.group(1).strip()
        if len(heading_text) >= 2 and any('\u0C00' <= c <= '\u0C7F' for c in heading_text):
            # Avoid duplicates
            if not any(h[0] == heading_text for h in headings):
                headings.append((heading_text, header_match.start(), header_match.end()))

    # Sort by position
    headings.sort(key=lambda x: x[1])

    if not headings:
        # No clear section headings, treat entire content as one section
        text = extract_text_from_html(post_body)
        if text.strip():
            # Try to extract first line as title
            lines = [l for l in text.split('\n') if l.strip()]
            title = lines[0][:50] if lines else "విభాగము"
            return [(title, text)]
        return []

    # Extract content between headings
    for i, (title, start_pos, end_pos) in enumerate(headings):
        # Content starts after current heading
        content_start = end_pos

        # Content ends at start of next heading or end of document
        if i + 1 < len(headings):
            content_end = headings[i + 1][1]
        else:
            content_end = len(content_html)

        # Extract HTML chunk
        section_html = content_html[content_start:content_end]

        # Parse and extract text
        section_soup = BeautifulSoup(section_html, 'lxml')
        content = extract_text_from_html(section_soup)

        if content.strip():
            sections.append((title, content))

    return sections


def extract_text_from_html(soup) -> str:
    """Extract clean text from HTML, preserving line breaks.

    Handles the blog's structure where each verse line is in a separate div.
    """
    # Replace br tags with newlines
    for br in soup.find_all('br'):
        br.replace_with('\n')

    # Replace divs with newlines (each div is typically a verse line)
    for div in soup.find_all('div'):
        # Add newline before div content
        div.insert_before('\n')

    text = soup.get_text()
    return clean_text(text)


def parse_blog_post(html: str, page_num: int) -> List[Tuple[str, str]]:
    """Parse a blog post page and extract all sections.

    Returns:
        List of (title, content) tuples
    """
    soup = BeautifulSoup(html, 'lxml')

    # Find post body - try multiple selectors
    post_body = soup.find('div', class_='post-body')
    if not post_body:
        post_body = soup.find('div', class_='entry-content')
    if not post_body:
        # Try to find any div with post content
        post_body = soup.find('div', {'itemprop': 'articleBody'})

    if not post_body:
        print(f"  WARNING: Could not find post body for page {page_num}")
        return []

    # Extract sections
    sections = extract_sections_from_post(post_body)

    return sections


def format_output(page_num: int, section_num: int, title: str, content: str) -> str:
    """Format the section content for saving to file."""
    output = []
    output.append("# గ్రంథము: పల్నాటివీరచరిత్ర")
    output.append("# రచయిత: శ్రీనాథభట్ట")
    output.append(f"# భాగము: {page_num:03d}")
    output.append(f"# శీర్షిక: {title}")
    output.append("")
    output.append(content)

    return '\n'.join(output)


def crawl_page(page_num: int, global_section_num: int) -> Tuple[int, int]:
    """Crawl a single blog post page and save its sections.

    Args:
        page_num: The page number (1-33)
        global_section_num: Current global section counter

    Returns:
        Tuple of (sections_saved, new_global_section_num)
    """
    url = get_url_for_page(page_num)
    print(f"\n  Page {page_num:02d}: Fetching...")

    html = fetch_page(url)
    if not html:
        print(f"    ERROR: Failed to fetch page {page_num}")
        return 0, global_section_num

    # Parse sections
    sections = parse_blog_post(html, page_num)

    if not sections:
        print(f"    WARNING: No sections found for page {page_num}")
        return 0, global_section_num

    # Save each section
    saved = 0
    for title, content in sections:
        # Format output
        output_text = format_output(page_num, global_section_num, title, content)

        # Create filename
        safe_title = sanitize_filename(title)
        filename = f"{global_section_num:03d}_{safe_title}.txt" if safe_title else f"{global_section_num:03d}.txt"
        filepath = OUTPUT_DIR / filename

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"    Section {global_section_num:03d}: {title[:40]}...")
        global_section_num += 1
        saved += 1

    return saved, global_section_num


def main():
    """Main function to crawl all blog posts."""
    print("=" * 60)
    print("పల్నాటివీరచరిత్ర (Palanati Veera Charitra) Crawler")
    print("=" * 60)
    print(f"Source: sahityasourabham.blogspot.com")
    print(f"Total pages to crawl: {TOTAL_PAGES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track statistics
    total_sections = 0
    global_section_num = 1

    for page_num in range(1, TOTAL_PAGES + 1):
        sections_saved, global_section_num = crawl_page(page_num, global_section_num)
        total_sections += sections_saved

        # Small delay between pages to be polite
        time.sleep(1)

    # Final summary
    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)
    print(f"Pages crawled: {TOTAL_PAGES}")
    print(f"Total sections saved: {total_sections}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Count files
    if OUTPUT_DIR.exists():
        file_count = len(list(OUTPUT_DIR.glob("*.txt")))
        print(f"Files created: {file_count}")


if __name__ == "__main__":
    main()
