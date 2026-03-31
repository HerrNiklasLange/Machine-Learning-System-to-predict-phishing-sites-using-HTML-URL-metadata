import pandas as pd
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

def main():
    df = pd.read_parquet(
        'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_preprocessed.parquet'
    )
    
    print(f"Loaded {len(df)} rows\n")
    
    # Build ML dataset
    print("--- Building ML Dataset ---")
    ml_df = build_ml_dataset(df)
    ml_df.to_parquet(
        'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_ml.parquet',
        index=False
    )
    print("ML dataset saved.\n")
    
    # Build DL dataset
    print("--- Building DL Dataset ---")
    dl_df = build_dl_dataset(df)
    dl_df.to_parquet(
        'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_dl.parquet',
        index=False
    )
    print("DL dataset saved.\n")
    
    # Final sanity check
    print("--- Sanity Check ---")
    print(f"ML dataset: {ml_df.shape}")
    print(f"ML Category distribution:\n{ml_df['Category'].value_counts()}")
    print(f"\nDL dataset: {dl_df.shape}")
    print(f"DL Category distribution:\n{dl_df['Category'].value_counts()}")


#Building ML dataset
def build_ml_dataset(df):
    print("Extracting URL features...")
    url_features = pd.DataFrame(
        df['url'].apply(extract_url_features).tolist()
    )
    
    print("Extracting HTML features (this may take a few minutes)...")
    html_features = pd.DataFrame(
        df['html'].apply(extract_html_features).tolist()
    )

    
    # Combine all features
    ml_df = pd.concat([
        df[['Category', 'collected', 'has_updated_date', 
            'has_status', 'has_emails', 'email_count','has_org',
            'has_country', 'has_registrar', 'has_name_server']],
        url_features,
        html_features
    ], axis=1)
    
    
    # Convert Category to binary
    ml_df['Category'] = (ml_df['Category'] == 'spam').astype(int)
    
    print(f"ML dataset shape: {ml_df.shape}")
    print(f"ML dataset columns: {ml_df.columns.tolist()}")
    return ml_df


#extracting URL features
def extract_url_features(url):
    try:
        parsed = urlparse(url if url.startswith('http') 
                         else 'http://' + url)
        return {
            'url_length': len(url),
            'domain_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_at': url.count('@'),
            'num_question_marks': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'num_digits': sum(c.isdigit() for c in url),
            'num_subdomains': max(len(parsed.netloc.split('.')) - 2, 0),
            'has_ip': 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0,
            'has_https': 1 if parsed.scheme == 'https' |'collected' == "old" else 0, #has historical data used https
            'has_port': 1 if parsed.port else 0,
        }
    except Exception:
        return {k: 0 for k in [
            'url_length', 'domain_length', 'path_length',
            'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_at', 'num_question_marks',
            'num_equals', 'num_ampersands', 'num_digits',
            'num_subdomains', 'has_ip', 'has_https', 'has_port'
        ]}

#Extract html features
def extract_html_features(html_str):
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        
        # Link analysis
        all_links = soup.find_all('a', href=True)
        total_links = len(all_links)
        external_links = sum(
            1 for a in all_links 
            if a['href'].startswith('http')
        )
        internal_links = total_links - external_links
        
        # Form analysis
        forms = soup.find_all('form')
        inputs = soup.find_all('input')
        password_inputs = soup.find_all('input', {'type': 'password'})
        
        # Script and resource analysis
        scripts = soup.find_all('script')
        external_scripts = sum(
            1 for s in scripts 
            if s.get('src', '').startswith('http')
        )
        iframes = soup.find_all('iframe')
        images = soup.find_all('img')
        
        # Meta and structural
        meta_tags = soup.find_all('meta')
        title = soup.find('title')
        
        return {
            'num_links': total_links,
            'num_external_links': external_links,
            'num_internal_links': internal_links,
            'ratio_external_links': external_links / max(total_links, 1),
            'num_forms': len(forms),
            'num_inputs': len(inputs),
            'has_password_input': 1 if password_inputs else 0,
            'num_scripts': len(scripts),
            'num_external_scripts': external_scripts,
            'num_iframes': len(iframes),
            'num_images': len(images),
            'num_meta_tags': len(meta_tags),
            'has_title': 1 if title else 0,
            'html_length': len(html_str),
        }
    except Exception:
        return {k: 0 for k in [
            'num_links', 'num_external_links', 'num_internal_links',
            'ratio_external_links', 'num_forms', 'num_inputs',
            'has_password_input', 'num_scripts', 'num_external_scripts',
            'num_iframes', 'num_images', 'num_meta_tags',
            'has_title', 'html_length'
        ]}

#building deep learn dataset
def build_dl_dataset(df):
    # For CNN - keep raw text, combine text columns
    dl_df = df[['Category', 'collected', 'html', 'url', 
                'registrar', 'name_servers', 'org', 
                'country', 'has_updated_date', 'status']].copy()
    
    # Combine metadata text into single field
    dl_df['metadata_text'] = (
        'registrar: ' + dl_df['registrar'].astype(str) + ' | ' +
        'nameservers: ' + dl_df['name_servers'].astype(str) + ' | ' +
        'org: ' + dl_df['org'].astype(str) + ' | ' +
        'country: ' + dl_df['country'].astype(str) + ' | ' +
        'status: ' + dl_df['status'].astype(str)
    )
    
    # Convert Category to binary
    dl_df['Category'] = (dl_df['Category'] == 'spam').astype(int)
    
    dl_df = dl_df[['Category', 'collected', 'url', 
                   'html', 'metadata_text', 'has_updated_date']]
    
    print(f"DL dataset shape: {dl_df.shape}")
    print(f"DL dataset columns: {dl_df.columns.tolist()}")
    return dl_df



if __name__ == "__main__":
    main()