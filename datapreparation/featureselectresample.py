"""Feature selection sul dataset resampled"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datapreparation import createstratifiedtestsetresampled

train_set,test_set = createstratifiedtestsetresampled.stratified_test_fraud_data()

train_set_labels1 = train_set["isFraud"].copy()
train_set_predictive=train_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)

test_set_labels1 = test_set["isFraud"].copy()
test_set_predictive=test_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)
# Definizione del Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
# Training
rnd_clf.fit(train_set_predictive, train_set_labels1)
# Feature evaluations
for name, score in zip(train_set_predictive.columns, rnd_clf.feature_importances_):
    print(name, score)

# Risultati feature selection
feat_importances = pd.Series(rnd_clf.feature_importances_, index=train_set_predictive.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()