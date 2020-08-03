"""Caricamento dataset"""
import pandas as pd


# Funzione che carica i dati dal dataset
def load_fraud_data():
    ABSOLUTE_PATH1 = "C:\\Users\\bernardino\\Desktop"
    FRAUD_PATH = ABSOLUTE_PATH1 + "\\FinancialDatasetForFraudDetection.csv"
    pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne dalla head()
    return pd.read_csv(FRAUD_PATH)