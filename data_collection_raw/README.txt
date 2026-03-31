README - data_collection_raw



This folder contains 4 data collection scripts.

SCRIPTS
-------

common_crawl.py
    Collects legitimate website URL and HTML data from Common Crawl.
    No configuration required. Once complete, move the output file
    to the data_new/ folder manually.

openphish_collector.py
    Collects phishing URL and HTML data from the OpenPhish live feed.
    No configuration required. Note: this script runs indefinitely
    and must be manually stopped once sufficient data has been collected.

whois_webscraper.py
    Retrieves WHOIS metadata for all collected URLs.
    Requires manual configuration before running:
        - Set the input file path to your merged dataset
        - Set the output/save file path to your desired location

reading_parquet_test.py
    A utility script for inspecting Parquet files.
    Written as a reference tool during development.
    Not required for the main pipeline.

RECOMMENDED RUN ORDER
----------------------
1. common_crawl.py
2. openphish_collector.py
3. whois_webscraper.py

NOTE
----
WHOIS collection is rate limited and may take several hours
depending on dataset size.