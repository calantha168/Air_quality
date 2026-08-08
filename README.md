# Air_quality

PM2.5 Prediction Using LSTM
This project predicts PM2.5 air pollution levels using an LSTM model implemented from scratch. It uses the Beijing Multi-Site Air Quality dataset, containing hourly measurements from 12 monitoring stations.

Files
        main.py — Runs the complete pipeline, including preprocessing, sequence generation, hyperparameter tuning, baseline              training, and ensemble training.
        lstm_model.py — Contains the LSTM implementation functions.
        preprocessing.py — Handles preprocessing.
        data_loader.py — Handles loading data.
        tuning_log.py — Logs hyperparameter tuning runs and results.
        experiment_log.py — Logs baseline and ensemble experiment runs and results.
        tuning_log.csv — Hyperparameter tuning results.
        experiment_log.csv — Baseline and ensemble experiment results.
        eda_notebook — Explores data.
        time_blocks.py — Creating definitions for time blocks.

    Dataset
    The model uses PM2.5 as the target variable and creates 24-hour sequences from the selected features.

    Running
    Run:
    python main.py

The program will load and preprocess the station data, create sequences, perform hyperparameter tuning, train the baseline and ensemble LSTM models, and save the experiment results.

﻿
