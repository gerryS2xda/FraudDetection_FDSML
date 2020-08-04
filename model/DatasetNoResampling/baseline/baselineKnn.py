"""Baseline K-NN"""
from getdata import fetchdata
from datapreparation import createstratifiedtestset
from visualizationresult import visualizeROC,visualizeconfusionmatrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


fraudata = fetchdata.load_fraud_data()
train_set,test_set = createstratifiedtestset.stratified_test_fraud_data()

train_set_labels1 = train_set["isFraud"].copy()
train_set_predictive=train_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)

test_set_labels1 = test_set["isFraud"].copy()
test_set_predictive=test_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)
# Definizione del K-NN Classifier
knn_clf=KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1)
knn_trained=knn_clf.fit(train_set_predictive, train_set_labels1)



# Creazione della ROC Curve
visualizeROC.create_roc_curve1(knn_trained,test_set_predictive,test_set_labels1)


y_pred = knn_clf.predict(test_set_predictive)
print(y_pred)

print("accuracy:")
print(accuracy_score(test_set_labels1,y_pred))
report=classification_report(y_pred, test_set_labels1)
print("Report:")
print(report)

# Creazione della Confusion Matrix
visualizeconfusionmatrix.confusion_matrix1(knn_clf,test_set_predictive,test_set_labels1,"KNN")

