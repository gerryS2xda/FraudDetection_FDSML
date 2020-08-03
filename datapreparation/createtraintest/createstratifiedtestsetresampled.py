"""Creazione di training set e test set sul dataset resampled"""
from sklearn.model_selection import StratifiedShuffleSplit
from Progetto.getdata.getdataandresampling import resampling


# Funzione che crea un training set e un test set stratificati
def stratified_test_fraud_data():
    fraudata_resampling = resampling.resamplingDataset()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(fraudata_resampling, fraudata_resampling["isFraud"]):
        strat_train_set = fraudata_resampling.iloc[train_index]
        strat_test_set = fraudata_resampling.iloc[test_index]

    return strat_train_set,strat_test_set


