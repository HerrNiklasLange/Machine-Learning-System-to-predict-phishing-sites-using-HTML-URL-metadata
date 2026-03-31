#This  merges all the data colected with all the datasets

#1 merges all the data
import pandas as pd
save_results_at = 'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_combined.parquet'
def main():
    df_old_main = pd.read_excel('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_old/df_main_old.xlsx') #replace path if relevant
    df_new_phishing_html = pd.read_parquet('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/phishing_openphish_html.parquet')
    df_new_legit_html = pd.read_parquet('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/commoncrawl_legit_CC-MAIN-2025-47.parquet')
    #C:\Users\nikla\OneDrive\Desktop\thesis-webscraper\data_new\commoncrawl_legit_CC-MAIN-2025-47.parquet
    df_new_legit_metadata = pd.read_excel('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/metadata.xlsx')
    df_new_phishing_metadata = pd.read_excel('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_new/metadataPhishing.xlsx')
    
    #adding to all new df column called collected 'new'
    df_new_phishing_html['collected'] = 'new'
    df_new_legit_html['collected'] = 'new'
    print(len(df_new_legit_html))
    print(len(df_new_phishing_html))

    #adding to all old df column called collected 'old'
    df_old_main['collected'] = 'old'
    



    # --- Merge new metadata with new html on URL ---
    df_new_phishing = pd.merge(
        df_new_phishing_html,
        df_new_phishing_metadata,
        on='url',
        how='left'
    )

    df_new_legit = pd.merge(
        df_new_legit_html,
        df_new_legit_metadata,
        on='url',
        how='left'
    )

    # --- Combine all into one dataframe ---
    df_combined = pd.concat(
        [df_old_main, df_new_phishing, df_new_legit],
        ignore_index=True
    )

    # Check
    print(f"Total rows: {len(df_combined)}")
    print(f"Category counts:\n{df_combined['Category'].value_counts()}")
    print(f"Collected counts:\n{df_combined['collected'].value_counts()}")
    print(f"Columns: {df_combined.columns.tolist()}")

    # --- Save merged dataset ---
    df_combined.to_parquet(save_results_at, index=False)
    print("Saved successfully.")

if __name__ == "__main__":
    main()

    #quick sanity check
    print("\nReloading Parquet...")

    df_parquet = pd.read_parquet(save_results_at)

    print("Parquet loaded successfully")
    print(df_parquet.head(2))
    print("\nSchema:")
    print(df_parquet.dtypes)

    print("\nDONE!")

    