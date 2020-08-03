"""Creazione di training set e test set sul dataset originale"""
from sklearn.model_selection import StratifiedShuffleSplit
from Progetto.getdata.getdataandresampling import fetchdata


# Funzione che crea un training set e un test set stratificati
def stratified_test_fraud_data():
    fraudata = fetchdata.load_fraud_data()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(fraudata, fraudata["isFraud"]):
        strat_train_set = fraudata.loc[train_index]
        strat_test_set = fraudata.loc[test_index]

    return strat_train_set,strat_test_set


