#!/usr/bin/env python3
"""
Crawler for బసవపురాణము (Basava Puranam) from Telugu Wikisource (te.wikisource.org)

Downloads all sections from 3 ఆశ్వాసములు and saves them as organized .txt files
in a nested directory structure.

Key features:
- 3 ఆశ్వాసములు, each on a separate page
- Each ఆశ్వాసము has multiple sections (విషయానుక్రమణిక)
- Content is extracted from poem divs and paragraph tags
- Requires SSL verification disabled (cert issue with Wikisource)
"""

import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Tuple, Optional
import time

from dwipada.data.crawl_base import (
    suppress_ssl_warnings, fetch_page, sanitize_filename,
    clean_html_content, clean_text, extract_section_content,
    find_content_div,
)
from dwipada.paths import DATA_DIR

suppress_ssl_warnings()

# Configuration
ASHVASAMS = [
    {
        "num": 1,
        "name": "ప్రథమాశ్వాసము",
        "url": "https://te.wikisource.org/wiki/బసవపురాణము/ప్రథమాశ్వాసము",
        "folder": "001_ప్రథమాశ్వాసము"
    },
    {
        "num": 2,
        "name": "ద్వితీయాశ్వాసము",
        "url": "https://te.wikisource.org/wiki/బసవపురాణము/ద్వితీయాశ్వాసము",
        "folder": "002_ద్వితీయాశ్వాసము"
    },
    {
        "num": 3,
        "name": "తృతీయాశ్వాసము",
        "url": "https://te.wikisource.org/wiki/బసవపురాణము/తృతీయాశ్వాసము",
        "folder": "003_తృతీయాశ్వాసము"
    },
]

OUTPUT_DIR = DATA_DIR / "basava_puranam"


def parse_ashvasam(html: str, ashvasam_name: str) -> List[Tuple[str, str]]:
    """Parse an ఆశ్వాసము page and extract all sections.

    Returns:
        List of (title, content) tuples
    """
    soup = BeautifulSoup(html, 'lxml')

    # Find main content area
    content_div = find_content_div(soup)
    if not content_div:
        print(f"  WARNING: Could not find main content div for {ashvasam_name}")
        return []

    # Get the HTML string of the content div for position-based extraction
    content_html = str(content_div)

    # Find section headings - Wikisource typically uses h2 or h3 for sections
    # Also check for centered divs with titles
    section_info = []  # List of (title, start_pos, end_pos_of_heading)

    # Method 1: Look for h2/h3 section headers
    for header in content_div.find_all(['h2', 'h3']):
        # Get the headline span inside
        headline = header.find('span', class_='mw-headline')
        if headline:
            title = headline.get_text(strip=True)
        else:
            title = header.get_text(strip=True)

        # Skip very short titles or navigation elements
        if len(title) < 2:
            continue
        # Skip table of contents header
        if title in ['విషయసూచిక', 'విషయానుక్రమణిక', 'Contents']:
            continue

        # Find position
        header_html = str(header)
        pos = content_html.find(header_html)
        if pos != -1:
            section_info.append((title, pos, pos + len(header_html)))

    # Method 2: If no h2/h3 found, look for centered div titles (like Sri Rama Parinayamu)
    if not section_info:
        for div in content_div.find_all('div', class_='tiInherit'):
            style = div.get('style', '')
            if 'text-align:center' in style or 'text-align: center' in style:
                p_tag = div.find('p')
                if p_tag:
                    title = p_tag.get_text(strip=True)
                    if len(title) < 3:
                        continue
                    # Skip book/chapter headers
                    if title == 'బసవపురాణము' or title == ashvasam_name:
                        continue

                    div_html = str(div)
                    pos = content_html.find(div_html)
                    if pos != -1:
                        section_info.append((title, pos, pos + len(div_html)))

    # Method 3: Look for bold centered paragraphs as section headers
    if not section_info:
        for p in content_div.find_all('p'):
            # Check if it looks like a heading (short, possibly bold)
            b_tag = p.find('b')
            if b_tag:
                title = b_tag.get_text(strip=True)
                if len(title) >= 3 and len(title) <= 100:
                    p_html = str(p)
                    pos = content_html.find(p_html)
                    if pos != -1:
                        section_info.append((title, pos, pos + len(p_html)))

    print(f"  Found {len(section_info)} sections in {ashvasam_name}")

    if not section_info:
        # If no sections found, treat the whole page as one section
        content = extract_section_content(content_div)
        if content.strip():
            return [(ashvasam_name, content)]
        return []

    # Extract content for each section based on positions
    sections = []

    for i, (title, start_pos, heading_end_pos) in enumerate(section_info):
        # Content starts after the heading
        content_start = heading_end_pos

        # Content ends at the start of the next heading (or end of content)
        if i + 1 < len(section_info):
            content_end = section_info[i + 1][1]
        else:
            content_end = len(content_html)

        # Extract the HTML chunk for this section
        section_html = content_html[content_start:content_end]

        # Parse and extract text
        content = extract_section_content(BeautifulSoup(section_html, 'lxml'))

        if content.strip():
            sections.append((title, content))

    return sections


def format_output(ashvasam_name: str, section_num: int, title: str, content: str) -> str:
    """Format the section content for saving to file."""
    output = []
    output.append("# గ్రంథము: బసవపురాణము")
    output.append(f"# ఆశ్వాసము: {ashvasam_name}")
    output.append(f"# విభాగము: {section_num:03d}")
    output.append(f"# శీర్షిక: {title}")
    output.append("")
    output.append(content)

    return '\n'.join(output)


def crawl_ashvasam(ashvasam: dict) -> int:
    """Crawl a single ఆశ్వాసము and save all its sections.

    Returns:
        Number of sections successfully saved
    """
    print(f"\n{'='*60}")
    print(f"Crawling {ashvasam['name']}")
    print(f"URL: {ashvasam['url']}")
    print(f"{'='*60}")

    # Create output folder
    output_folder = OUTPUT_DIR / ashvasam['folder']
    output_folder.mkdir(parents=True, exist_ok=True)

    # Fetch the page
    print("  Fetching page...")
    html = fetch_page(ashvasam['url'])
    if not html:
        print(f"  ERROR: Failed to fetch page for {ashvasam['name']}")
        return 0

    print(f"  Page fetched successfully ({len(html)} bytes)")

    # Parse sections
    print("  Parsing sections...")
    sections = parse_ashvasam(html, ashvasam['name'])

    if not sections:
        print(f"  WARNING: No sections found for {ashvasam['name']}")
        return 0

    # Save each section
    print(f"  Saving {len(sections)} sections...")
    success_count = 0

    for i, (title, content) in enumerate(sections, 1):
        # Format output
        output_text = format_output(ashvasam['name'], i, title, content)

        # Create filename
        safe_title = sanitize_filename(title)
        filename = f"{i:03d}_{safe_title}.txt" if safe_title else f"{i:03d}.txt"
        filepath = output_folder / filename

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output_text)

        success_count += 1
        print(f"    Section {i:03d}: {title[:40]}...")

    return success_count


def main():
    """Main function to crawl all ఆశ్వాసములు."""
    print("=" * 60)
    print("బసవపురాణము (Basava Puranam) Crawler")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total ఆశ్వాసములు: {len(ASHVASAMS)}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track statistics
    total_sections = 0

    for ashvasam in ASHVASAMS:
        sections_saved = crawl_ashvasam(ashvasam)
        total_sections += sections_saved
        print(f"  {ashvasam['name']}: {sections_saved} sections saved")

        # Small delay between pages to be polite
        time.sleep(1)

    # Final summary
    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)
    print(f"Total sections saved: {total_sections}")
    print(f"Output directory: {OUTPUT_DIR}")

    # List output folders
    print("\nOutput folders:")
    for ashvasam in ASHVASAMS:
        folder = OUTPUT_DIR / ashvasam['folder']
        if folder.exists():
            file_count = len(list(folder.glob("*.txt")))
            print(f"  {ashvasam['folder']}: {file_count} files")


if __name__ == "__main__":
    main()
