README - data_merged



This folder contains 4 data files that are the output of the 
preprocessing pipeline.

FILES
-------


df_combined.parquet
    The raw merged dataset combining both the historical and modern
    data before any preprocessing has been done.

df_preprocessed.parquet
    The combined dataset after preprocessing has been applied.
    This includes dropped rows, cleaned HTML and binary metadata flags.
    This is the main dataset used to generate the ML and DL datasets.

df_ml.parquet
    The preprocessed dataset with feature engineering applied.
    Contains the 38 engineered numerical features across URL, HTML
    and metadata categories. Used to train RF, LR and KNN models.

df_dl.parquet
    The preprocessed dataset kept as raw text for the CNN model.
    Contains raw URL strings, raw HTML text and concatenated
    metadata text. Used to train the CNN model.

NOTE
----
df_ml.parquet and df_dl.parquet are kept separate to prevent
data leakage between the feature engineering and deep learning
approaches. Do not merge them.