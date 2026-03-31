import pandas as pd
#some data overview and see if there are any obvious mistakes and data that needs to be removed
df = pd.read_parquet('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_combined.parquet')

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print()

# Check what URL and HTML look like
#print("Sample URL:")
#print(df['url'].iloc[0])
#print()
#print("Sample HTML (first 200 chars):")
#print(str(df['html'].iloc[0])[:200])
#print()

# Check date columns
print("Sample creation_date:")
print(df['creation_date'].iloc[0])
print(type(df['creation_date'].iloc[0]))


print("Sample")
print(df.sample(5, random_state=42).apply(
    lambda x: str(x)[:150]).to_string())

#general shape of the data
print(f"Total rows: {len(df)}")
print(f"Category counts:\n{df['Category'].value_counts()}")
print(f"Collected counts:\n{df['collected'].value_counts()}")
print(f"Columns: {df.columns.tolist()}")