import pandas as pd

PATH1='C:\dataset'
ABSOLUTE_PATH1 = 'C:/Users/gerar/PycharmProjects/FraudDetection_FDSML/'
FRAUD_DSPATH = ABSOLUTE_PATH1 + "dataset/FinancialDatasetForFraudDetection.csv"

def load_fraud_data(fraud_path = FRAUD_DSPATH):
    return pd.read_csv(fraud_path)

#Call function
fraudata = load_fraud_data()
fraudata.info()
print(fraudata.head(10)) #necessario perche' PyCharm non e' Python Shell


