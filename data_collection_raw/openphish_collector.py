import time
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd



# OpenPhish collector loops for as long as possible
openphish_url_feed = "https://openphish.com/feed.txt"
output_saved_at = "C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/phishing_openphish_html.parquet"
sleep_time = 1800  # reruns every 30 minutes
max_size_html = 1_000_000  # this is in place just to not accidently download a truly massive amounf of data
request_timeout = 10

Headers = {
    "User-Agent": "Academic-Phishing-Research/1.0",
    "Accept": "text/html,application/xhtml+xml"
}

#OpenPhish invinite run collector
def run_collector():
    print("Infinity loop started")

    seen_urls = load_existing_urls()

    while True:
        try:
            urls = fetch_openphish_urls()
            #checking if new URL is relevant
            new_urls = [u for u in urls if u not in seen_urls]

            print(f" New URLs: {len(new_urls)}")

            rows = []

            for url in new_urls:
                html = fetch_html(url)
                if html:
                    rows.append({
                        "url": url,
                        "html": html,
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "openphish"
                    })
                    seen_urls.add(url)

                time.sleep(1)

            save_rows(rows)
            print(f" Saved {len(rows)} pages")

        except Exception as e:
            print(f"[WARN] Error: {e}")
        pd.read_parquet("phishing_openphish_html.parquet").head()
        print(f" Sleeping {sleep_time // 60} minutes\n")
        time.sleep(sleep_time)

#Load existing so no additional repeats
def load_existing_urls():
    if Path(output_saved_at).exists():
        #print("Read Correctly")
        df = pd.read_parquet(output_saved_at, columns=["url"])
        return set(df["url"].astype(str))
    return set()

#Save file to parquet - CSV does not work
def save_rows(rows):
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if Path(output_saved_at).exists():
        old_df = pd.read_parquet(output_saved_at)
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset="url")
    else:
        df = new_df

    df.to_parquet(output_saved_at, index=False)

#fetching openphish data
def fetch_openphish_urls():
    resp = requests.get(openphish_url_feed, timeout=request_timeout)
    resp.raise_for_status()
    return [u.strip() for u in resp.text.splitlines() if u.strip()]

#HTML collector using request and beautiful soup
def fetch_html(url):
    try:
        r = requests.get(
            url,
            headers=Headers,
            timeout=request_timeout,
            allow_redirects=True
        )

        if "text/html" not in r.headers.get("Content-Type", "").lower():
            return None

        if len(r.content) > max_size_html:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        return str(soup)

    except Exception:
        return None




if __name__ == "__main__":
    run_collector()
    #import os
    #cwd = os.getcwd()
    #print(cwd)
    #df = pd.read_parquet("data_new/phishing_openphish_html.parquet")
    #print(df.head())
    #print(df.count())
   