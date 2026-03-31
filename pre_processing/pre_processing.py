import pandas as pd
import ast
import re

#2 some cleaning to the HTML preproccessing
def preprocess(df):

    # dropping rows where domain is missing
    df = df.dropna(subset=['domain'])
    print(f"Rows after dropping missing domains: {len(df)}")

    # dropping columns
    df = df.drop(columns=['creation_date', 'expiration_date', 
                           'timestamp', 'source'])

    # converting to binary 
    df['has_updated_date'] = df['updated_date'].notna().astype(int)
    df['has_emails'] = df['emails'].notna().astype(int)
    df['has_status'] = df['status'].notna().astype(int)
    df['has_org'] = df['org'].notna().astype(int)
    df['has_country'] = df['country'].notna().astype(int)
    df['has_name_server'] = df['name_servers'].notna().astype(int)
    df['has_registrar'] = df['registrar'].notna().astype(int)
    # email counter
    df['email_count'] = df['emails'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )


    # --- Drop original columns now replaced by binary ---
    df = df.drop(columns=['updated_date', 'emails'])

    # --- Fill remaining text columns with empty string for CNN ---
    text_cols = ['name_servers', 'org', 'country', 'registrar']
    df[text_cols] = df[text_cols].fillna('NA')

    # NA check + general check
    print(f"Missing values after preprocessing:")
    print(df.isnull().sum())
    print(f"\n Category counts:\n{df['Category'].value_counts()}")
    print(f"\n Old|New counts:\n{df['collected'].value_counts()}")
    print(f"\n Final columns: {df.columns.tolist()}")
    print(f"\n Final shape: {df.shape}")

   
    return df
#cleaning the htmls
def clean_html(html_str):
    #formating issues like "['<html>', '', '<body>']" need to be resolved
    try:
        # Check if it looks like a stringified list
        if html_str.startswith("<!DOCTYPE html>'") or "', '" in html_str[:50]:
            # parsing and joining
            try:
                parsed = ast.literal_eval(f"['{html_str}']")
                return ' '.join(parsed)
            except:
                # cleaining
                cleaned = re.sub(r"'(,\s*'|\s*$)", '', html_str)
                cleaned = cleaned.replace("', '", ' ').replace("'", '')
                return cleaned
        else:
            # Normal HTML - return as is
            return html_str
    except Exception:
        return html_str
    

def main():
    df = pd.read_parquet('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_combined.parquet')
    
    df = preprocess(df)
    print("Cleaning HTML column...")
    df['html'] = df['html'].apply(clean_html)

    # Verify fix
    weird_after = df['html'].str.contains("', '", na=False).sum()
    print(f"Weird list-like HTML rows after cleaning: {weird_after}")

    # Check a previously weird row looks sensible
    print("\nSample cleaned row:")
    print(df['html'].iloc[0][:200])
    #-----------------------------
    # Source - https://stackoverflow.com/a/66657167
    # Posted by dallonsi, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-03-27, License - CC BY-SA 4.0
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #-------------------------------

    # Save preprocessed dataset
    df.to_parquet('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_preprocessed.parquet', index=False)
    print("\nSaved successfully.")
    print(df.head())
    
if __name__ == "__main__":
    main()