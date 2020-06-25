from getdata import fetchdata
from matplotlib import pyplot as plt
from datapreparation import createstratifiedtestset
import pandas as pd
import numpy as np


#Call function
print('Loading data...')
fraudata = fetchdata.load_fraud_data()

print('Head:')
print(fraudata.head(10)) #necessario perche' PyCharm non e' Python Shell
print('Info:')
fraudata.info(null_counts=True) #
print('Categorical:')
print(fraudata["type"].value_counts())
print('Describe:')
print(fraudata.describe())
plt.figure(1)
print('Istogramma amount:')

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

fraudata["isFraud"].hist(bins=50,figsize=(10,5))#bins e' il numero di valori differenti
plt.title("isFraud")
plt.figure(4)

fraudata["isFlaggedFraud"].hist(bins=50,figsize=(10,5))#bins e' il numero di valori differenti
plt.title("isFlaggedFraud")
plt.show()
print("Creo test set stratificato")
train_set,test_set= createstratifiedtestset.stratified_test_fraud_data()
print("TF/TD e TN/TD sull'intero ds")
print(fraudata["isFraud"].value_counts() / len(fraudata))
print("TF/TD e TN/TD  sul train_set")
print(train_set["isFraud"].value_counts() / len(train_set))
print("TF/TD e TN/TD  sul test_set")
print(test_set["isFraud"].value_counts() / len(test_set))
correlation_ma=fraudata.corr()
print("Correlazione:isFraud")
print(correlation_ma["isFraud"].sort_values(ascending=False)) #isFraud è molto correlato con amount e isFlaggedFraud
print("Correlazione:isFlaggedFraud")
print(correlation_ma["isFlaggedFraud"].sort_values(ascending=False)) #isFlaggedFraud è molto correlato con isFraud


