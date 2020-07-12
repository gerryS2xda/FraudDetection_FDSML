from getdata import fetchdata
from datapreparation import createstratifiedtestset
from visualizationresult import visualizeROC,visualizeconfusionmatrix

import numpy as np

import matplotlib.pyplot as plt


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve

fraudata = fetchdata.load_fraud_data()
train_set,test_set = createstratifiedtestset.stratified_test_fraud_data()

train_set_labels1 = train_set["isFraud"].copy()
train_set_predictive=train_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)

test_set_labels1 = test_set["isFraud"].copy()
test_set_predictive=test_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)
knn_clf=KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1)
knn_trained=knn_clf.fit(train_set_predictive, train_set_labels1)




visualizeROC.create_roc_curve1(knn_trained,test_set_predictive,test_set_labels1)


y_pred = knn_clf.predict(test_set_predictive)
print(y_pred)

print("accuracy:")
print(accuracy_score(test_set_labels1,y_pred))
report=classification_report(y_pred, test_set_labels1)
print("Report:")
print(report)

visualizeconfusionmatrix.confusion_matrix1(knn_clf,test_set_predictive,test_set_labels1,"KNN")

train_sizes, train_scores, test_scores = learning_curve(knn_clf,
                                                        train_set_predictive,
                                                        train_set_labels1,
                                                        # Number of folds in cross-validatio
                                                        # Evaluation metric
                                                        cv=2,
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        shuffle=True,
                                                        n_jobs=-1,
                                                        random_state=42,
                                                        # 50 different sizes of the training s
                                                        train_sizes=np.linspace(0.01, 1.0,1000))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()