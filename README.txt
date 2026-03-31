README - User Guide

Prereq
-------------------
The following software and libraries are required to run this project.

- Python 3.9.13
- All dependencies can be installed by running:

    pip install -r requirements.txt

Projects structure
-----------------

thesis/
├── data_collection_raw/   # Raw data collection scripts
├── data_merged/           # Merged historical and modern dataset
├── data_new/              # Modern collected dataset
├── data_old/              # Historical dataset
├── DL_model/              # CNN model scripts
├── ML_model/              # ML model scripts (RF, LR, KNN)
├── models/                # Saved trained models
├── plots/                 # Generated evaluation plots
├── pre_processing/        # Preprocessing and feature engineering
├── requirements.txt       # Required dependencies

Note: Each folder contains a README.txt file with further instructions
specific to that stage of the pipeline. All file paths in the scripts
are absolute and will need to be updated to match your local directory
structure before running.

Running the Pipeline
-------
The pipeline should be run in the following order:

1. Run data collection scripts in data_collection_raw/
2. Run preprocessing scripts in pre_processing/
3. Train ML models by running scripts in ML_model/
4. Train CNN model by running scripts in DL_model/

Expected Output
----
- Trained models saved to models/
- Evaluation plots saved to plots/
- Results tables generated automatically as CSV files

Known Issues
---------------
- WHOIS rate limiting may significantly slow metadata collection
- CNN training is CPU-bound and may take several hours
- A checkpoint file is saved every 1,000 pages during data
  collection to prevent data loss