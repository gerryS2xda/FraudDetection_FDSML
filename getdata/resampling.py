import pandas as pd
from sklearn.utils import resample
from getdata import fetchdata

def resamplingDataset():
    fraudata = fetchdata.load_fraud_data()
    df_class_1_over = fraudata[fraudata["isFraud"]==1].sample(3000000, random_state=42,replace=True)
    df_0=resample(fraudata[fraudata["isFraud"]==0],n_samples=3000000,replace=True,random_state=42)
    df_resampled = pd.concat([df_0, df_class_1_over], axis=0)
    return df_resampled
    