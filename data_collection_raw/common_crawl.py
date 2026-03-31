import gzip
import io
import random
import requests
from warcio.archiveiterator import ArchiveIterator
import pandas as pd
import os

crawl_id = "CC-MAIN-2025-47" #the common craw version that will be collected
n_pages = 8000
warc_max_attempts = 30
common_crawl_url = "https://data.commoncrawl.org/crawl-data"
output_saved_at = f"commoncrawl_legit_{crawl_id}.parquet"

def main():
    warc_paths = get_warc_paths(crawl_id)
    random.shuffle(warc_paths)
    
    collected = []
    pages_count = 0
    warcs_used = 0

    for path in warc_paths:
        if pages_count >= n_pages or warcs_used >= warc_max_attempts:
            break
            
        warc_url = f"https://data.commoncrawl.org/{path}"
        warcs_used += 1
        
        try:
            for url, html_bytes in iter_records_from_warc(warc_url):
                html = html_bytes.decode("utf-8", errors="ignore")
                
                # Basic quality filter - skip very short pages
                if len(html) < 500:
                    continue
                    
                collected.append({"url": url, "html": html})
                pages_count += 1
                
                # Save checkpoint every 1000 pages
                # so you don't lose everything if it crashes
                if pages_count % 1000 == 0:
                    print(f" Checkpoint: {pages_count} pages collected...")
                    checkpoint_df = pd.DataFrame(collected)
                    checkpoint_df.to_parquet(
                        f"checkpoint_{pages_count}.parquet", 
                        index=False
                    )
                
                if pages_count >= n_pages:
                    break
                    
        except Exception as e:
            print(f"Error reading WARC {warc_url}: {e}")
            continue
            
        print(f"Collected so far: {pages_count} pages")

    print(f"Finished. Collected {pages_count} pages "
          f"from {warcs_used} WARC files.")
    
    if collected:
        df = pd.DataFrame(collected)
        
        print(f"\nTotal rows: {len(df)}")
        print(f"Sample URLs:")
        print(df['url'].head(5).tolist())
        
        # Save final parquet
        df.to_parquet(output_saved_at, index=False)
        print(f"Saved to {output_saved_at}")
        print(f"Full path: {os.path.realpath(output_saved_at)}")
        
        # Clean up checkpoints
        for f in os.listdir('.'):
            if f.startswith('checkpoint_') and f.endswith('.parquet'):
                os.remove(f)
                print(f"Removed checkpoint: {f}")
    else:
        print("No pages collected. Check network / settings.")

def get_warc_paths(crawl_id: str):
    paths_url = f"{common_crawl_url}/{crawl_id}/warc.paths.gz"
    print(f" Downloading WARC paths from {paths_url}")
    resp = requests.get(paths_url, timeout=60)
    resp.raise_for_status()
    warc_paths = []
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        for line in gz:
            warc_paths.append(line.decode("utf-8").strip())
    print(f" Found {len(warc_paths)} WARC files in this crawl.")
    return warc_paths

def iter_records_from_warc(warc_url: str):
    print(f" Streaming WARC: {warc_url}")
    with requests.get(warc_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for record in ArchiveIterator(r.raw, arc2warc=True):
            if record.rec_type != "response":
                continue
            http_headers = record.http_headers
            if not http_headers:
                continue
            content_type = (
                http_headers.get_header("Content-Type") or ""
            ).lower()
            if "text/html" not in content_type:
                continue
            url = record.rec_headers.get_header("WARC-Target-URI")
            if not url:
                continue
            try:
                payload = record.content_stream().read()
            except Exception:
                continue
            if not payload:
                continue
            yield url, payload


if __name__ == "__main__":
    main()