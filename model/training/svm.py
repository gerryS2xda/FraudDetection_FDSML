from datapreparation import preparetraintest
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from visualizationresult import visualizeconfusionmatrix
from visualizationresult import visualizeROC

train_set_preparate,test_set,train_set_labels1,train_set_labels2=preparetraintest.prepared_fraud_data()
train_p=train_set_preparate
train_l= train_set_labels1
test_set_labels1 = test_set["isFraud"].copy()
test_set_predictive=test_set.drop(["type","nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)

#training svc kernel linear


clf = svm.SVC(kernel='linear', C=5,probability=True) #SVC = classificatore SVM
proba1=clf.fit(train_p, train_l) #X (attr. predittivi - dati training), y valori attr. dipendenti

#roc curve svc kernel lineare

visualizeROC.create_roc_curve1(proba1,test_set_predictive,test_set_labels1)


#predizioni svc lineare

predizioni2=proba1.predict(test_set_predictive)
print("accuracy:")
print(accuracy_score(predizioni2,test_set_labels1))
report=classification_report(predizioni2, test_set_labels1)
print("Report:")
print(report)

#calcolo matrice di confusione svc lineare

visualizeconfusionmatrix.confusion_matrix1(clf, test_set_predictive, test_set_labels1, "SVC linear")




predizioni2=proba1.predict(test_set_predictive)
print("accuracy:")
print(accuracy_score(predizioni2,test_set_labels1))
report=classification_report(predizioni2, test_set_labels1)
print("Report:")
print(report)

#training svc kernel non linear

clf1 = svm.SVC(kernel="poly", degree=5, coef0=1, C=5,probability=True) #SVC = classificatore SVM
proba2=clf1.fit(train_p, train_l)


#roc curve svc kernel non lineare

visualizeROC.create_roc_curve1(proba2,test_set_predictive,test_set_labels1)

#predizioni svc non lineare

predizioni4=proba2.predict(test_set_predictive)
print("accuracy:")
print(accuracy_score(predizioni4,test_set_labels1))
report1=classification_report(predizioni4, test_set_labels1)
print("Report:")
print(report1)

#calcolo matrice di confusione svc non lineare

visualizeconfusionmatrix.confusion_matrix1(clf1, test_set_predictive, test_set_labels1, "SVC non-linear")


