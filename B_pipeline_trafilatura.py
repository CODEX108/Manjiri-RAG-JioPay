# pipeline_b_trafilatura.py
import time, json, re, logging
import trafilatura
from trafilatura import sitemaps
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.WARNING)
SEEDS = ["https://www.jio.com/business/"]

def tokenize(s):
    return 0 if not s else len(re.findall(r"\w+", s))

def fallback_extract_links_from_page(url, max_links=500):
    """Simple fallback: fetch the page and extract hrefs (same-host only)."""
    html = trafilatura.fetch_url(url)
    if not html:
        return []
    hrefs = set()
    for m in re.findall(r'href=["\'](.*?)["\']', html, flags=re.IGNORECASE):
        if not m or m.startswith("mailto:") or m.startswith("tel:"):
            continue
        abs_url = urljoin(url, m.split("#")[0])
        parsed = urlparse(abs_url)
        if not parsed.scheme.startswith("http"):
            continue
        # keep same host as seed root (avoid external domains)
        if urlparse(url).netloc == parsed.netloc:
            hrefs.add(abs_url)
        if len(hrefs) >= max_links:
            break
    return list(hrefs)

def crawl(seeds, max_pages=200):
    seen = set()
    results = []
    t0 = time.time()
    for seed in seeds:
        # 1) primary: use sitemap_search (returns page URLs) — per docs
        try:
            page_urls = sitemaps.sitemap_search(seed) or []
        except Exception as e:
            logging.warning("sitemap_search failed for %s: %s", seed, e)
            page_urls = []

        # 2) fallback: extract links from homepage if sitemap_search returned nothing
        if not page_urls:
            logging.warning("No sitemap URLs found for %s — using page-link fallback", seed)
            page_urls = fallback_extract_links_from_page(seed)

        for u in page_urls:
            if len(results) >= max_pages:
                break
            if u in seen:
                continue
            seen.add(u)

            downloaded = trafilatura.fetch_url(u)
            if not downloaded:
                results.append({
                    "url": u, "status": None, "error": "fetch_failed",
                    "tokens": 0, "noise_ratio": None
                })
                continue

            extracted = trafilatura.extract(downloaded, include_tables=False, include_links=False)
            if not extracted:
                results.append({
                    "url": u, "status": 200, "error": "no_main_content",
                    "tokens": 0, "noise_ratio": None
                })
                continue

            tokens = tokenize(extracted)
            noise_ratio = 1 - (len(extracted) / max(len(downloaded), 1))
            results.append({
                "url": u,
                "status": 200,
                "tokens": tokens,
                "noise_ratio": round(noise_ratio, 3)
            })

    elapsed = time.time() - t0
    return results, elapsed

if __name__ == "__main__":
    results, elapsed = crawl(SEEDS)
    ok = [r for r in results if r.get("status") == 200 and r.get("tokens", 0) > 0]
    report = {
        "pipeline": "trafilatura",
        "pages_total": len(results),
        "pages_ok": len(ok),
        "tokens_total": sum(r["tokens"] for r in ok),
        "avg_noise_ratio": round(sum(r["noise_ratio"] for r in ok) / max(len(ok), 1), 3) if ok else None,
        "throughput_pages_per_sec": round(len(results) / max(elapsed, 1e-6), 2),
        "failures": [r for r in results if r.get("status") != 200 or r.get("tokens", 0) == 0]
    }
    print(json.dumps(report, indent=2))
