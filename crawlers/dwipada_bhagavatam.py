#!/usr/bin/env python3
"""
Crawler for Dwipada Bhagavatam from Telugu Wikisource (te.wikisource.org)

Downloads all chapters across 3 kandas, formats verses as couplets,
and saves one file per kanda.

Key features:
- 3 kandas (books): Madhura, Kalyana, Jagadabhiraksha
- Verses formatted as couplets (2-line pairs)
- Section headings preserved
- SSL verification disabled (Wikisource cert issue)

Usage:
    python dwipada_bhagavatam.py
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional

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
BASE_URL = "https://te.wikisource.org/wiki/ద్విపదభాగవతము/"

KANDAS = [
    {
        "name": "MadhuraKanda",
        "telugu": "మధురాకాండము",
        "url_slug": "మధురాకాండము",
        "filename": "1_madhurakanda.txt",
    },
    {
        "name": "KalyanaKanda",
        "telugu": "కల్యాణకాండము",
        "url_slug": "కల్యాణకాండము",
        "filename": "1_kalyanakanda.txt",
    },
    {
        "name": "JagadabhirakshaKanda",
        "telugu": "జగదభిరక్షకాండము",
        "url_slug": "జగదభిరక్షకాండము",
        "filename": "1_jagadabhirakshakanda.txt",
    },
]

OUTPUT_DIR = DATA_DIR / "dwipada_bhagavatam2"


def clean_verse_line(line: str) -> Optional[str]:
    """Clean a single verse line.

    Returns None if the line should be skipped (ellipsis-only, empty, etc).
    """
    line = line.strip()
    if not line:
        return None

    # Skip ellipsis-only lines
    if re.match(r'^[\.…\s]+$', line):
        return None

    # Remove footnote markers like [1], [2]
    line = re.sub(r'\[\d+\]', '', line)

    # Remove parentheses but keep the text inside: (text) -> text
    line = re.sub(r'\(([^)]*)\)', r'\1', line)
    # Remove any remaining unmatched parens
    line = line.replace('(', '').replace(')', '')
    # Remove quotation marks, exclamation marks, semicolons, commas
    line = re.sub(r'["""\u201c\u201d!;,]', '', line)

    # Remove trailing page numbers (e.g. "text 10", "text;20")
    # These are page markers from the print edition, typically multiples of 10
    line = re.sub(r'\d{1,3}\s*$', '', line)

    # Clean up extra spaces
    line = re.sub(r'\s+', ' ', line).strip()

    if not line:
        return None

    return line


def format_couplets(lines: List[str]) -> str:
    """Format verse lines as couplets (pairs of 2 lines with blank line between)."""
    # Clean all lines, filtering out None
    cleaned = []
    for line in lines:
        result = clean_verse_line(line)
        if result:
            cleaned.append(result)

    if not cleaned:
        return ""

    # Group into pairs (couplets)
    couplets = []
    for i in range(0, len(cleaned), 2):
        if i + 1 < len(cleaned):
            couplets.append(f"{cleaned[i]}\n{cleaned[i+1]}")
        else:
            # Odd line at the end - include it standalone
            couplets.append(cleaned[i])

    return "\n\n".join(couplets)


def parse_kanda_page(html: str) -> List[Tuple[str, str]]:
    """Parse a kanda page and extract headings and verse blocks.

    Returns:
        List of (type, text) tuples where type is "heading" or "verses"
    """
    soup = BeautifulSoup(html, 'lxml')

    # Find main content area
    content_div = find_content_div(soup)

    if not content_div:
        print("  WARNING: Could not find main content div")
        return []

    # Clean common noise elements
    clean_html_content(content_div)

    result = []

    # Process poem divs - they contain both headings and verses
    poems = content_div.find_all('div', class_='poem')

    for poem in poems:
        # Check for embedded heading(s) inside the poem
        headings_in_poem = poem.find_all('div', class_='tiInherit')

        if not headings_in_poem:
            # No headings - the whole poem is verse content
            for br in poem.find_all('br'):
                br.replace_with('\n')
            text = poem.get_text()
            lines = [l for l in text.split('\n') if l.strip()]
            if lines:
                result.append(("verses", lines))
        else:
            # Poem contains heading(s) - need to separate heading from verses
            # Process the poem's HTML content in order
            _extract_from_poem_with_headings(poem, result)

    return result


def _extract_from_poem_with_headings(poem, result: list):
    """Extract headings and verses from a poem div that contains tiInherit headings."""
    # First, replace <br> with newlines
    for br in poem.find_all('br'):
        br.replace_with('\n')

    current_verses = []

    for child in poem.children:
        if hasattr(child, 'get') and child.get('class') and 'tiInherit' in child.get('class', []):
            # This is a heading
            # First, flush any accumulated verses
            if current_verses:
                result.append(("verses", current_verses))
                current_verses = []

            bold = child.find('b')
            if bold:
                heading_text = bold.get_text(strip=True)
                # Skip the kanda title itself (e.g. "ద్విపదభాగవతము")
                if heading_text and 'ద్విపదభాగవతము' not in heading_text:
                    result.append(("heading", heading_text))
        elif hasattr(child, 'get_text'):
            text = child.get_text()
            lines = [l for l in text.split('\n') if l.strip()]
            current_verses.extend(lines)
        elif isinstance(child, str):
            text = child.strip()
            if text:
                lines = [l for l in text.split('\n') if l.strip()]
                current_verses.extend(lines)

    # Flush remaining verses
    if current_verses:
        result.append(("verses", current_verses))


def build_output(parsed: List[Tuple[str, str]]) -> str:
    """Build the final output string from parsed headings and verse blocks."""
    output_parts = []

    for item_type, content in parsed:
        if item_type == "heading":
            # Clean punctuation from headings too
            clean_heading = re.sub(r'["""\u201c\u201d!;,]', '', content).strip()
            output_parts.append(f"\n# {clean_heading}\n")
        elif item_type == "verses":
            formatted = format_couplets(content)
            if formatted:
                output_parts.append(formatted)

    # Join and clean up excessive blank lines
    text = "\n".join(output_parts)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip() + "\n"


def crawl_kanda(kanda: dict, output_dir: Path) -> bool:
    """Crawl a kanda and save as a single file.

    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"Crawling {kanda['name']} ({kanda['telugu']})")
    print(f"{'='*60}")

    # Build URL
    url = BASE_URL + kanda['url_slug']
    print(f"  URL: {url}")

    # Fetch the page
    html = fetch_page(url)
    if not html:
        print(f"  ERROR: Failed to fetch kanda page")
        return False

    # Parse headings and verses
    parsed = parse_kanda_page(html)

    heading_count = sum(1 for t, _ in parsed if t == "heading")
    verse_count = sum(1 for t, _ in parsed if t == "verses")
    print(f"  Found {heading_count} chapter headings, {verse_count} verse blocks")

    if not parsed:
        print(f"  WARNING: No content found")
        return False

    # Build output
    output_text = build_output(parsed)

    # Save to file
    filepath = output_dir / kanda['filename']
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output_text)

    line_count = output_text.count('\n')
    print(f"  Saved: {filepath} ({line_count} lines)")
    return True


def main():
    """Main function to crawl all kandas."""
    print("=" * 60)
    print("Dwipada Bhagavatam Crawler")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total kandas: {len(KANDAS)}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track statistics
    success_count = 0

    for kanda in KANDAS:
        if crawl_kanda(kanda, OUTPUT_DIR):
            success_count += 1

    # Final summary
    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)
    print(f"Kandas saved: {success_count}/{len(KANDAS)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
