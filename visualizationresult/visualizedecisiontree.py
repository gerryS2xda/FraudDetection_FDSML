"""Visualizza l'albero di decisione"""
from sklearn import tree
from matplotlib import pyplot as plt

# Funzione che visualizza l'albero di decisione
def decision_tree(clf):
    plt.figure(3)
    tree.plot_tree(clf)
