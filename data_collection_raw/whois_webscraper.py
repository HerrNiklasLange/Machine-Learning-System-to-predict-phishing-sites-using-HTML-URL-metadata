import pandas as pd
import whois
import tldextract
import time




#Change the following input where relavant
read_at = "C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/commoncrawl_legit_CC-MAIN-2025-47.parquet" 
output_file_at = "/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/metadata.xlsx"
url_column = "url"   

def main():
    print("Loading Excel file...")
    #df = pd.read_excel(read_at)
    #df = pd.read_csv(read_at)
    df = pd.read_parquet(read_at)

    results = []
    x = 0
    for idx, url in enumerate(df[url_column]):
        print(f"\n[{idx+1}/{len(df)}] Processing:", url)

        # Extract root domain
        domain = extract_domain(url)
        print("→ Extracted domain:", domain)

        if domain is None:
            print("→ Skipped (could not extract domain)")
            results.append({})
            continue

        # Query WHOIS
        w = safe_whois(domain)
        
        # Append cleaned fields
        results.append({
            "url": url,
            "domain": domain,
            "creation_date": w.get("creation_date"),
            "expiration_date": w.get("expiration_date"),
            "updated_date": w.get("updated_date"),
            "registrar": w.get("registrar"),
            "name_servers": w.get("name_servers"),
            "org": w.get("org"),
            "country": w.get("country"),
            "status": w.get("status"),
            "emails": w.get("emails"),
        })
        #Left over code for testing if the the data pipelines works before running it for a long period of time
        #x = x + 1
        # Avoid rate limits / temporary blocks
        #if x >100:
        #    print("\n Saving results...")
        #    out_df = pd.DataFrame(results)
        #    out_df = out_df.applymap(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
        #    out_df.to_excel(output_file_at, index=False)
        #    print(f"FINISHED! Results saved to {output_file_at}")
        #    exit()
        time.sleep(0.5)
       
    print("\n Saving results...")
    out_df = pd.DataFrame(results)
    out_df = out_df.applymap(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
    out_df.to_excel(output_file_at, index=False)
    print(f"FINISHED! Results saved to {output_file_at}")

# Extracting domains from the URL to ensure better reading and no missinput for the WHOIS URL
def extract_domain(url):
    ext = tldextract.extract(url)

    if ext.domain and ext.suffix:

        return ext.domain + "." + ext.suffix
    return None

#WhoIS look up with error handling exceptions
def safe_whois(domain):
    try:
        return whois.whois(domain)
    
    except Exception as e:

        print(f" Could not fetch WHOIS for {domain}: {e}")
        
        return {}



if __name__ == "__main__":
    main()
