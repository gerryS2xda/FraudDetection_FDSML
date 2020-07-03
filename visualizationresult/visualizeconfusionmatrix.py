from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#Crea la matrice di confusione
def confusion_matrix1(clf,test_set,test_labels,name):  #name= nome del classificatore

    titles_options = [("Confusion matrix, without normalization "+name, None), ("Normalized confusion matrix "+name, 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_set,test_labels ,cmap=plt.cm.Blues,normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
        plt.show()

