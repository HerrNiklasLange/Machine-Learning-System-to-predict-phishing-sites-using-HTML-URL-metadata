
import pandas as pd

#First time using parquet and checking format
print(pd.read_parquet("phishing_openphish_html.parquet").head)
print(len(pd.read_parquet("phishing_openphish_html.parquet")))

