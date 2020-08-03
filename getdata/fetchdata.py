"""Caricamento dataset"""
import pandas as pd
import io
import requests

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"


# Funzione che carica i dati dal dataset
def load_fraud_data(url):
    s = requests.get(url).content
    pd.set_option('display.max_columns', None)
    return pd.read_csv(io.StringIO(s.decode('utf-8')))
