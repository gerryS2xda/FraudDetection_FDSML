from datapreparation import createstratifiedtestsetresampled

from sklearn.preprocessing import StandardScaler


def prepared_fraud_data():
    print("Creo test set stratificato")
    train_set,test_set = createstratifiedtestsetresampled.stratified_test_fraud_data()
    train_set_labels1 = train_set["isFraud"].copy()
    test_set_labels1= test_set["isFraud"].copy()
    train_set_predictive = train_set.drop(["type", "nameOrig", "nameDest", 'isFraud', 'isFlaggedFraud', 'step', 'newbalanceDest', 'oldbalanceDest'],axis=1)
    test_set_predictive = test_set.drop( ["type", "nameOrig", "nameDest", 'isFraud', 'isFlaggedFraud', 'newbalanceDest', 'oldbalanceDest','step'],axis=1)
    print("Training set stratificato dopo feature selection:")
    print(train_set_predictive)
    num_pipeline = StandardScaler()
    train_set_preparate = num_pipeline.fit_transform(train_set_predictive)
    test_set_preparate = num_pipeline.transform(test_set_predictive)
    return train_set_preparate,test_set_preparate,train_set_labels1,test_set_labels1

