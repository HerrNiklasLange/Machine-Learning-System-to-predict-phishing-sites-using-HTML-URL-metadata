import pandas as pd
# this is more for fun than anything else with some initial exploration, was not used in the final report
df = pd.read_parquet('C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_ml.parquet')


print("Collected vs Label")
collected_label = pd.crosstab(
    df['collected'], 
    df['label'],
    margins=True
)
print(collected_label)

print("\nAs percentages:")
collected_pct = pd.crosstab(
    df['collected'], 
    df['label'],
    normalize='index'
).round(3) * 100
print(collected_pct)


print("\n Top 15 Countries vs Label")
top_countries = df['country'].value_counts().nlargest(15).index
df_top = df[df['country'].isin(top_countries)]

country_label = pd.crosstab(
    df_top['country'],
    df_top['label'],
    margins=True
).sort_values(1, ascending=False)
print(country_label)

print("\nAs percentages (% spam per country):")
country_pct = pd.crosstab(
    df_top['country'],
    df_top['label'],
    normalize='index'
).round(3) * 100
print(country_pct.sort_values(1, ascending=False))


print("\n Top 15 Countries vs Collected (old/new)")
country_collected = pd.crosstab(
    df_top['country'],
    df_top['collected']
).sort_values('old', ascending=False)
print(country_collected)