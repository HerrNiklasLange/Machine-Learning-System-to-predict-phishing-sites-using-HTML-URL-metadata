README - data_new

This folder contains the raw modern dataset files collected from
Common Crawl and OpenPhish, along with their metadata.


Files
--------------


commoncrawl_legit_CC_MAIN_2025_47.csv / .parquet
    Legitimate URL and HTML data collected from Common Crawl.
    Both CSV and Parquet versions are provided.

phishing_openphish_html.parquet
    Phishing URL and HTML data collected from OpenPhish.

metadata.xlsx
    WHOIS metadata for the legitimate Common Crawl entries.

metadataPhishing.xlsx
    WHOIS metadata for the phishing OpenPhish entries.

intial_checks_df_new.py
    Initial data inspection script used during development.
    Not required for the main pipeline.