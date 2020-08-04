"""Decision tree fine-tuned"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from datapreparation import preparetraintest
from visualizationresult import visualizeROC, visualizeconfusionmatrix

train_set,test_set,train_labels,test_labels=preparetraintest.prepared_fraud_data()
# Definizione del Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=4,random_state=42,min_samples_leaf= 100)
decision_tree_trained=tree_clf.fit(train_set, train_labels)

export_graphviz(tree_clf, out_file="decisiontreetuning.dot",
                feature_names=['amount','oldbalanceOrg','newbalanceDest'],
                class_names=['0','1'],
                rounded=True, filled=True
                )


# Creazione della ROC Curve
visualizeROC.create_roc_curve1(decision_tree_trained,test_set,test_labels)


y_pred = tree_clf.predict(test_set)
print(y_pred)

print("accuracy:")
print(accuracy_score(test_labels,y_pred))
report=classification_report(y_pred, test_labels)
print("Report:")
print(report)

# Creazione della Confusion Matrix
visualizeconfusionmatrix.confusion_matrix1(tree_clf,test_set,test_labels,"Decision Tree")

