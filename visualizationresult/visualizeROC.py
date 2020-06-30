from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def create_roc_curve1(trainedclassifier,test_set,test_set_labels):
    predizioni1 = trainedclassifier.predict_proba(test_set)
    fpr, tpr, thresholds = roc_curve(test_set_labels, predizioni1[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)