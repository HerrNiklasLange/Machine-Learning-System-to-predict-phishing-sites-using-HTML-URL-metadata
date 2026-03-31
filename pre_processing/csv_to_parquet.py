import pandas as pd
import base64
from pathlib import Path

#this is a converted from csv to parquet as initial phishing data was saved as CSV with signifanc loading and saving erros

#Note the data was moved to each relavant location and an error would appear now 
read_csv_at = r"/Users/nikla/phishing_openphish_html.csv"
parquet_file = r"/Users/nikla/phishing_openphish_html.parquet"


print("Loading CSV...")

df = pd.read_csv(
    read_csv_at,
    engine="python",   # robust CSV parsing
    sep=",",           # change to ";" if you used semicolon
    quoting=0
)

print("CSV loaded")
print(df.head(2))
print("\nColumns:", df.columns.tolist())
print("Rows:", len(df))

def decode_html(x):
    try:
        return base64.b64decode(x).decode("utf-8", errors="ignore")
    except Exception:
        return ""

print("\nDecoding HTML...")
df["html_decoded"] = df["html"].astype(str).apply(decode_html)

print("HTML decoded sample:")
print(df["html_decoded"].iloc[0][:500])


print("\nSaving Parquet...")
df.to_parquet(parquet_file, index=False)
print(f"Saved to {parquet_file}")


print("\nReloading Parquet...")

df_parquet = pd.read_parquet(parquet_file)

#conversion check
print("Parquet loaded successfully")
print(df_parquet.head(2))
print(df_parquet.dtypes)

print("\nFINISHED!Parquet pipeline verified")
