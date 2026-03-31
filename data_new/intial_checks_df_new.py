import pandas as pd

df = pd.read_parquet(
    'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_combined.parquet'
)

# Filter to new data only
df_new = df[df['collected'] == 'new'].copy()

#checks
print("Sample URLs:")
print(df_new['url'].sample(5, random_state=42).to_string())
print()

#sample character
print("Sample HTML (first 150 chars):")
print(df_new['html'].sample(5, random_state=42).apply(
    lambda x: str(x)[:150]).to_string())

#shape and distribution
print(f"Total rows: {len(df_new)}")
print(f"\nCategory distribution:\n{df_new['Category'].value_counts()}")
print(f"\nColumns: {df_new.columns.tolist()}")