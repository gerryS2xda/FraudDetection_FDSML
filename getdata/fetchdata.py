import pandas as pd


ABSOLUTE_PATH1 = 'C:/Users/gerar/PycharmProjects/FraudDetection_FDSML/'
ABSOLUTE_PATH2 = 'C:/Users/bernardino/PycharmProjects/FraudDetection_FDSML/'
FRAUD_DSPATH = ABSOLUTE_PATH2 + "dataset/FinancialDatasetForFraudDetection.csv"

def load_fraud_data(fraud_path = FRAUD_DSPATH):
    return pd.read_csv(fraud_path)





