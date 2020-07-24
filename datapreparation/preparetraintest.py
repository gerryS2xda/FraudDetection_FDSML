from datapreparation import createstratifiedtestset

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from getdata import fetchdata

def prepared_fraud_data():
    fraudata = fetchdata.load_fraud_data()
    print("Creo test set stratificato")
    train_set,test_set = createstratifiedtestset.stratified_test_fraud_data()
    train_set_labels1 = train_set["isFraud"].copy()
    test_set_labels1= test_set["isFraud"].copy()
    train_set_predictive = train_set.drop(["type", "nameOrig", "nameDest", 'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'step', 'oldbalanceDest'],axis=1)
    test_set_predictive = test_set.drop( ["type", "nameOrig", "nameDest", 'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'step', 'oldbalanceDest'],axis=1)
    print("Training set stratificato dopo feature selection:")
    print(train_set_predictive)
    print("TF/TD e TN/TD sull'intero ds") #TF->tuple Fraud ,TD->tuple ds,TN->tuple notFraud
    print(fraudata["isFraud"].value_counts() / len(fraudata))
    print("TF/TD e TN/TD  sul train_set")
    print(train_set["isFraud"].value_counts() / len(train_set))
    print("TF/TD e TN/TD  sul test_set")
    print(test_set["isFraud"].value_counts() / len(test_set))
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    train_set_preparate = num_pipeline.fit_transform(train_set_predictive)
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    test_set_preparate = num_pipeline.fit_transform(test_set_predictive)

    return train_set_preparate,test_set_preparate,train_set_labels1,test_set_labels1

