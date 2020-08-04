"""Caricamento dataset"""
import pandas as pd


ABSOLUTE_PATH1 = 'C:/Users/gerar/PycharmProjects/FraudDetection_FDSML/'
ABSOLUTE_PATH2 = 'C:/Users/bernardino/PycharmProjects/FraudDetection_FDSML/'
FRAUD_DSPATH = ABSOLUTE_PATH2 + "dataset/FinancialDatasetForFraudDetection.csv"

# Funzione che carica i dati dal dataset
def load_fraud_data(fraud_path = FRAUD_DSPATH):
    print('Caricamento dei dati...')
    pd.set_option('display.max_columns', None)
    return pd.read_csv(fraud_path)





