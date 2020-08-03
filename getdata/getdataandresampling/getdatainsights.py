"""Reperimento di informazioni dettagliate del dataset"""
from Progetto.getdata.getdataandresampling import fetchdata
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# Caricamento dei dati
print('Caricamento dei dati...')
fraudata = fetchdata.load_fraud_data()

# Data profiling rapido
print('Head:')
print(fraudata.head(10)) #necessario perche' PyCharm non e' Python Shell

print('Info:')
fraudata.info(null_counts=True)

print('Categorical:')
print(fraudata["type"].value_counts())

print('Describe:')
print(fraudata.describe())


print('Istogramma amount:')
plt.figure(1)
fraudata["amount_cat"]=pd.cut(fraudata["amount"],bins=[0,1000,10000,50000,100000,200000,500000,1000000,5000000,np.inf],labels=[0,1,2,3,4,5,6,7,8])
fraudata["amount_cat"].hist(bins=50,figsize=(10,5))#bins e' il numero di valori differenti
plt.title("amount_cat")
print("Statistiche delle categorie")
print(fraudata["amount_cat"].value_counts())
plt.figure(2)

print('Istogramma type:')
fraudata["type"].hist(bins=50,figsize=(10,5))#bins e' il numero di valori differenti
plt.title("type")
print("Statistiche delle categorie")
print(fraudata["type"].value_counts())
plt.figure(3)

print('Istogramma isFraud:')
fraudata["isFraud"].hist(bins=50,figsize=(10,5))#bins e' il numero di valori differenti
plt.title("isFraud")
plt.figure(4)

print('Istogramma isFlaggedFraud:')
fraudata["isFlaggedFraud"].hist(bins=50,figsize=(10,5))#bins e' il numero di valori differenti
plt.title("isFlaggedFraud")
plt.show()





