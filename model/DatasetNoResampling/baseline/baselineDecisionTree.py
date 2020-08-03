"""Baseline Decision Tree"""
from getdata import fetchdata
from datapreparation import createstratifiedtestset
from visualizationresult import visualizeROC,visualizeconfusionmatrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


fraudata = fetchdata.load_fraud_data()
train_set,test_set = createstratifiedtestset.stratified_test_fraud_data()

train_set_labels1 = train_set["isFraud"].copy()
train_set_predictive=train_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)

test_set_labels1 = test_set["isFraud"].copy()
test_set_predictive=test_set.drop(['type',"nameOrig","nameDest",'isFraud','isFlaggedFraud'],axis=1)
# Definizione del Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=3,random_state=42)
decision_tree_trained=tree_clf.fit(train_set_predictive, train_set_labels1)

export_graphviz(tree_clf, out_file="decisiontreebaseline.dot",
    feature_names=['step','amount','oldbalanceOrg','newbalanceOrg','oldbalanceDest','newbalanceDest'],
    class_names=['0','1'],
    rounded=True, filled=True
)

path = 'decisiontreebaseline.dot'
s = Source.from_file(path)
s.view()

# Creazione della ROC Curve
visualizeROC.create_roc_curve1(decision_tree_trained,test_set_predictive,test_set_labels1)


y_pred = tree_clf.predict(test_set_predictive)
print(y_pred)

print("accuracy:")
print(accuracy_score(test_set_labels1,y_pred))
report=classification_report(y_pred, test_set_labels1)
print("Report:")
print(report)

# Creazione della Confusion Matrix
visualizeconfusionmatrix.confusion_matrix1(tree_clf,test_set_predictive,test_set_labels1,"Decision Tree")

