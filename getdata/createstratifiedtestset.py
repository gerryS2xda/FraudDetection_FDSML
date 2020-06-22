from sklearn.model_selection import StratifiedShuffleSplit
from getdata import fetchdata

def stratified_test_fraud_data():
    # Call function
    print('Loading data...')
    fraudata = fetchdata.load_fraud_data()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(fraudata, fraudata["isFraud"]):
        strat_train_set = fraudata.loc[train_index]
        strat_test_set = fraudata.loc[test_index]

    return strat_train_set,strat_test_set


