README - pre_processing


This folder contains all preprocessing and feature engineering scripts.
These should be run after data collection and before model training.

Script - Recommeded run order
---------

1. merge_data.py
   Merges historical and modern datasets and adds collection
   labels (old/new) and class labels (spam/ham).

2. pre_processing.py
   Cleans HTML, drops missing domain rows and creates binary
   metadata flags. Outputs df_preprocessed.parquet.

3. Feature_engineering_DL_ML_df_creation.py
   Applies feature engineering to create df_ml.parquet and
   df_dl.parquet. Both saved to data_merged/.

Utility scripts
-------

csv_to_parquet.py       
    Converts CSV files to Parquet format
data_exploration.py     
    Exploratory data analysis scripts
data_overview.py        
    Prints dataset shape and class distribution