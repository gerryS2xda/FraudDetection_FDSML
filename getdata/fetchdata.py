import pandas as pd


ABSOLUTE_PATH1 = 'C:/Users/gerar/PycharmProjects/FraudDetection_FDSML/'
ABSOLUTE_PATH2 = 'C:/Users/bernardino/PycharmProjects/FraudDetection_FDSML/'
FRAUD_DSPATH = ABSOLUTE_PATH2 + "dataset/FinancialDatasetForFraudDetection.csv"

def load_fraud_data(fraud_path = FRAUD_DSPATH):
    pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne dalla head()
    return pd.read_csv(fraud_path)





