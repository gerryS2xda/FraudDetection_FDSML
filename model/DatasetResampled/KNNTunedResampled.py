"""K-NN fine-tuned sul dataset resampled"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from datapreparation import preparetraintestresampled
from visualizationresult import visualizeROC, visualizeconfusionmatrix

train_set,test_set,train_labels,test_labels= preparetraintestresampled.prepared_fraud_data()

knn_clf=KNeighborsClassifier(n_neighbors=3,metric="euclidean",n_jobs=-1)
knn_trained=knn_clf.fit(train_set, train_labels)

# Creazione della ROC Curve
visualizeROC.create_roc_curve1(knn_trained, test_set, test_labels)


y_pred = knn_clf.predict(test_set)
print(y_pred)

print("accuracy:")
print(accuracy_score(test_labels,y_pred))
report=classification_report(y_pred, test_labels)
print("Report:")
print(report)

# Creazione della ROC Curve
visualizeconfusionmatrix.confusion_matrix1(knn_clf, test_set, test_labels, "KNN")



