from datapreparation import createstratifiedtestset
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from getdata import fetchdata

fraudata = fetchdata.load_fraud_data()
print("Creo test set stratificato")
train_set,test_set = createstratifiedtestset.stratified_test_fraud_data()
train_set_labels1 = train_set["isFraud"].copy()
train_set_labels2= train_set["isFlaggedFraud"].copy()
train_set_copy=train_set.drop(['isFraud','isFlaggedFraud'],axis=1)
print("Training set stratificato:")
print(train_set_copy)
print("TF/TD e TN/TD sull'intero ds") #TF->tuple Fraud ,TD->tuple ds,TN->tuple notFraud
print(fraudata["isFraud"].value_counts() / len(fraudata))
print("TF/TD e TN/TD  sul train_set")
print(train_set["isFraud"].value_counts() / len(train_set))
print("TF/TD e TN/TD  sul test_set")
print(test_set["isFraud"].value_counts() / len(test_set))

'''encoder = LabelEncoder()
train_set_cat=train_set_copy["type"]
train_set_cat_encoded = encoder.fit_transform(train_set_cat)
encoder1 = OneHotEncoder()
train_set_cat1 = encoder1.fit_transform(train_set_cat_encoded.reshape(-1,1))
print(train_set_cat1)
'''

train_set_copy_num=train_set_copy.drop(["type","nameOrig","nameDest"],axis=1)
num_pipeline = Pipeline([('std_scaler', StandardScaler())])
housing_num_tr = num_pipeline.fit_transform(train_set_copy_num)
