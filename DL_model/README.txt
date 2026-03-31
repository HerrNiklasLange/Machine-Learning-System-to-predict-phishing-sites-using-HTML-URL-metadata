README - DL_model


This folder contains the deep learning model script.

FILES
------------------------

CNN.py
    Trains the three-branch CNN model on the DL dataset.
    Input: df_dl.parquet from data_merged/
    Output: saves best model to models/cnn_best.pt
    Note: Training is CPU-bound and over 30 minutes.