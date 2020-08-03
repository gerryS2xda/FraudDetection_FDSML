"""Creazione del nuovo dataset mediante strategie di resampling"""
import pandas as pd
from sklearn.utils import resample

from Progetto.getdata.getdataandresampling import fetchdata


# Funzione che crea il resampled dataset
def resamplingDataset():
    fraudata = fetchdata.load_fraud_data()
    df_class_0_under = fraudata[fraudata["isFraud"] == 0].sample(3000000, random_state=42)
    df_1_over = resample(fraudata[fraudata["isFraud"] == 1], n_samples=3000000, random_state=42)
    df_resampled = pd.concat([df_class_0_under, df_1_over], axis=0)
    return df_resampled
