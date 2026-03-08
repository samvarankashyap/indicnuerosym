#!/usr/bin/env python3
"""
Crawler for Ranganatha Ramayanam from andhrabharati.com

Downloads all 405 chapters across 7 kandas and saves them as organized .txt files.
"""

import re
import time
import json
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional, Tuple, List, Dict

from dwipada.paths import DATA_DIR
from dwipada.data.crawl_base import (
    suppress_ssl_warnings,
    fetch_page as _base_fetch_page,
    sanitize_filename,
    clean_html_content,
    clean_text,
    find_content_div,
)

# Configuration
BASE_URL = "https://www.andhrabharati.com/itihAsamulu/RanganathaRamayanamu/"

KANDAS = [
    {"name": "BalaKanda", "telugu": "బాలకాండము", "chapters": 31, "folder": "01_BalaKanda"},
    {"name": "AyodhyaKanda", "telugu": "అయోధ్యాకాండము", "chapters": 35, "folder": "02_AyodhyaKanda"},
    {"name": "AranyaKanda", "telugu": "అరణ్యకాండము", "chapters": 28, "folder": "03_AranyaKanda"},
    {"name": "KishkindhaKanda", "telugu": "కిష్కింధాకాండము", "chapters": 25, "folder": "04_KishkindhaKanda"},
    {"name": "SundaraKanda", "telugu": "సుందరకాండము", "chapters": 27, "folder": "05_SundaraKanda"},
    {"name": "YuddhaKanda", "telugu": "యుద్ధకాండము", "chapters": 170, "folder": "06_YuddhaKanda"},
    {"name": "UttaraKanda", "telugu": "ఉత్తరకాండము", "chapters": 89, "folder": "07_UttaraKanda"},
]

OUTPUT_DIR = DATA_DIR / "ranganatha_ramayanam"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# Request settings
REQUEST_DELAY = 1.5  # seconds between requests
MAX_RETRIES = 3
TIMEOUT = 30


def fetch_page(url: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Fetch HTML content from a URL with retry logic.

    Uses base fetch_page with verify_ssl=True and custom timeout.
    """
    return _base_fetch_page(url, retries=retries, verify_ssl=True, timeout=TIMEOUT)


def extract_footnotes(soup: BeautifulSoup) -> List[str]:
    """Extract footnotes from the page.

    Returns:
        List of footnote texts
    """
    footnotes = []

    # Find the fnlist div which contains footnotes
    fnlist = soup.find('div', class_='fnlist')
    if fnlist:
        # Each footnote starts with ↑ symbol
        fn_text = fnlist.get_text(separator='\n')
        for line in fn_text.split('\n'):
            line = line.strip()
            if line.startswith('↑'):
                # Remove the ↑ and add the footnote
                footnotes.append(line[1:].strip())
            elif line and footnotes:
                # Continuation of previous footnote
                footnotes[-1] += ' ' + line

    return footnotes


def extract_content(html: str) -> Tuple[str, str, List[str]]:
    """Extract main content and footnotes from HTML.

    Returns:
        Tuple of (title, content, footnotes)
    """
    soup = BeautifulSoup(html, 'lxml')

    # Extract title from chapter_hdr div
    title = ""
    chapter_hdr = soup.find('div', class_='chapter_hdr')
    if chapter_hdr:
        title = chapter_hdr.get_text(strip=True)
    else:
        # Fallback to title tag
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean up - remove site name
            if '-' in title:
                title = title.split('-')[0].strip()

    # Extract footnotes before modifying DOM
    footnotes = extract_footnotes(soup)

    # Find main content in wmsect div
    wmsect = soup.find('div', class_='wmsect')
    if not wmsect:
        # Fallback to body if wmsect not found
        wmsect = soup.find('body')

    if wmsect:
        # Remove elements we don't want
        for tag in wmsect.find_all(['script', 'style']):
            tag.decompose()

        # Remove navigation links div
        for nav in wmsect.find_all('div', class_='chapter_links'):
            nav.decompose()

        # Remove fnlist (footnotes - we already extracted them)
        for fn in wmsect.find_all('div', class_='fnlist'):
            fn.decompose()

        # Remove chapter_hdr (we already got the title)
        for hdr in wmsect.find_all('div', class_='chapter_hdr'):
            hdr.decompose()

        # Replace <br> tags with newline markers before text extraction
        for br in wmsect.find_all('br'):
            br.replace_with('\n')

        # Remove superscript footnote numbers by unwrapping (keeps surrounding text together)
        for sup in wmsect.find_all('sup'):
            sup.decompose()

        # Remove anchor tags but keep their non-footnote text
        for a in wmsect.find_all('a'):
            href = a.get('href', '')
            # If it's a footnote reference, just remove it
            if href.startswith('#fn_'):
                a.decompose()
            else:
                # For other links, unwrap to keep text
                a.unwrap()

        # Get text content - don't use separator to avoid breaking text at inline elements
        text = wmsect.get_text()

        # Now normalize the whitespace while preserving intentional line breaks
        # Split by newlines, strip each line, rejoin
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
    else:
        text = ""

    # Clean up the text
    text = clean_text(text)

    # Remove any remaining footnote/appendix markers
    # Remove inline [A], [B], etc. markers
    text = re.sub(r'\[[A-Za-z]\]', '', text)

    # Remove standalone digit lines (footnote markers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just numbers (footnote markers)
        if stripped and stripped.isdigit():
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Clean up again after removing footnote markers
    text = clean_text(text)

    return title, text, footnotes


def format_output(kanda_telugu: str, chapter_num: int, title: str, content: str,
                  footnotes: List[str]) -> str:
    """Format the chapter content for saving to file."""
    output = []
    output.append(f"# కాండము: {kanda_telugu}")
    output.append(f"# అధ్యాయము: {chapter_num:03d}")
    output.append(f"# శీర్షిక: {title}")
    output.append("")
    output.append(content)

    # Add footnotes if any
    if footnotes:
        output.append("")
        output.append("---")
        output.append("పాదసూచికలు (Footnotes):")
        for i, fn in enumerate(footnotes, 1):
            output.append(f"[{i}] {fn}")

    return '\n'.join(output)


def load_checkpoint() -> Dict:
    """Load the checkpoint file to resume from where we left off."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": []}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def crawl_chapter(kanda: Dict, chapter_num: int, output_folder: Path) -> bool:
    """Crawl a single chapter and save it to a file.

    Returns:
        True if successful, False otherwise
    """
    # Build URL
    url = f"{BASE_URL}RanganathaRamayanamu_{kanda['name']}_{chapter_num:03d}.html"

    # Fetch the page
    html = fetch_page(url)
    if not html:
        print(f"  ERROR: Failed to fetch {url}")
        return False

    # Extract content
    title, content, footnotes = extract_content(html)

    if not content.strip():
        print(f"  WARNING: No content extracted from {url}")
        return False

    # Format output
    output_text = format_output(
        kanda['telugu'], chapter_num, title, content, footnotes
    )

    # Create filename
    safe_title = sanitize_filename(title)
    filename = f"{chapter_num:03d}_{safe_title}.txt" if safe_title else f"{chapter_num:03d}.txt"
    filepath = output_folder / filename

    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output_text)

    return True


def crawl_kanda(kanda: Dict, checkpoint: Dict) -> int:
    """Crawl all chapters of a kanda.

    Returns:
        Number of chapters successfully downloaded
    """
    output_folder = OUTPUT_DIR / kanda['folder']
    output_folder.mkdir(parents=True, exist_ok=True)

    success_count = 0

    print(f"\n{'='*60}")
    print(f"Crawling {kanda['name']} ({kanda['telugu']})")
    print(f"Chapters: {kanda['chapters']}")
    print(f"{'='*60}")

    for chapter_num in range(1, kanda['chapters'] + 1):
        chapter_id = f"{kanda['name']}_{chapter_num:03d}"

        # Skip if already completed
        if chapter_id in checkpoint.get('completed', []):
            print(f"  Chapter {chapter_num:03d}: Already downloaded, skipping...")
            success_count += 1
            continue

        print(f"  Chapter {chapter_num:03d}/{kanda['chapters']:03d}: Downloading...")

        if crawl_chapter(kanda, chapter_num, output_folder):
            success_count += 1
            checkpoint['completed'].append(chapter_id)
            save_checkpoint(checkpoint)
            print(f"  Chapter {chapter_num:03d}: Done")
        else:
            print(f"  Chapter {chapter_num:03d}: FAILED")

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    return success_count


def main():
    """Main function to crawl all kandas."""
    print("="*60)
    print("Ranganatha Ramayanam Crawler")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total kandas: {len(KANDAS)}")
    print(f"Total chapters: {sum(k['chapters'] for k in KANDAS)}")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"Previously completed: {len(checkpoint.get('completed', []))} chapters")

    # Track statistics
    total_success = 0
    total_chapters = sum(k['chapters'] for k in KANDAS)

    try:
        for kanda in KANDAS:
            success = crawl_kanda(kanda, checkpoint)
            total_success += success
            print(f"\n{kanda['name']}: {success}/{kanda['chapters']} chapters downloaded")
    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved to checkpoint.")
        print("Run the script again to resume from where you left off.")

    # Final summary
    print("\n" + "="*60)
    print("CRAWL COMPLETE")
    print("="*60)
    print(f"Total chapters downloaded: {total_success}/{total_chapters}")
    print(f"Output directory: {OUTPUT_DIR}")

    # List failed chapters if any
    completed_set = set(checkpoint.get('completed', []))
    failed = []
    for kanda in KANDAS:
        for chapter_num in range(1, kanda['chapters'] + 1):
            chapter_id = f"{kanda['name']}_{chapter_num:03d}"
            if chapter_id not in completed_set:
                failed.append(chapter_id)

    if failed:
        print(f"\nFailed chapters ({len(failed)}):")
        for chapter_id in failed[:10]:  # Show first 10
            print(f"  - {chapter_id}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
