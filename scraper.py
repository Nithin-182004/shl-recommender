"""SHL Catalog Scraper — fetches Individual Test Solutions from the product catalog."""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def scrape_page(start=0, session=None):
    """Scrape a single catalog page starting at the given offset."""
    url = f"https://www.shl.com/products/product-catalog/?start={start}&type=1"
    print(f"  Fetching start={start}...")

    try:
        resp = session.get(url, headers=HEADERS, timeout=60)
        print(f"  Status: {resp.status_code}, Length: {len(resp.text)}")
        if resp.status_code != 200:
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    for link in soup.find_all("a"):
        href = link.get("href", "")
        if "/products/product-catalog/view/" not in href:
            continue

        name = link.get_text(strip=True)
        if not name or len(name) < 2:
            continue

        full_url = href if href.startswith("http") else "https://www.shl.com" + href

        row = link.find_parent("tr")
        remote = ""
        adaptive = ""
        test_type = ""

        if row:
            tds = row.find_all("td")
            for i, td in enumerate(tds):
                text = td.get_text(strip=True)
                if re.match(r'^[ABCDEKPS]$', text):
                    test_type = text
                if i == 1 and not remote:
                    spans = td.find_all("span")
                    if spans:
                        classes = " ".join(spans[0].get("class", []))
                        if "check" in classes.lower() or "yes" in text.lower():
                            remote = "Yes"
                        elif "cross" in classes.lower() or "no" in text.lower():
                            remote = "No"
                    if not remote:
                        remote = text
                if i == 2 and not adaptive:
                    spans = td.find_all("span")
                    if spans:
                        classes = " ".join(spans[0].get("class", []))
                        if "check" in classes.lower() or "yes" in text.lower():
                            adaptive = "Yes"
                        elif "cross" in classes.lower() or "no" in text.lower():
                            adaptive = "No"
                    if not adaptive:
                        adaptive = text
                if i == 3 and not test_type:
                    test_type = text

        items.append({
            "name": name,
            "url": full_url,
            "test_type": test_type,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
        })

    return items


def main():
    print("=" * 60)
    print("SHL Individual Test Solutions Scraper")
    print("=" * 60)

    session = requests.Session()
    all_items = []

    for page in range(32):
        start = page * 12
        items = scrape_page(start, session)

        if not items:
            print(f"  No items on page {page+1}, stopping.")
            break

        all_items.extend(items)
        print(f"  Got {len(items)} items (total: {len(all_items)})")
        time.sleep(1.5)

    seen = set()
    unique = []
    for item in all_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    print(f"\n{len(unique)} unique assessments scraped.")

    os.makedirs("data", exist_ok=True)
    with open("data/catalog.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"Saved to data/catalog.json")


if __name__ == "__main__":
    main()
