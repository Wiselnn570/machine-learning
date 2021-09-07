import numpy as np
from sklearn.datasets import load_wine
import graphviz
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
wine = load_wine()
feature_name = wine.feature_names
target_name = wine.target_names
x = wine.data
y = wine.target
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=0)

clf = tree.DecisionTreeClassifier(criterion='entropy'
                                    ,splitter='random'
                                    ,random_state=30
                                    ,max_depth=3
                                    )
clf.fit(xtrain,ytrain)
print(clf.predict(xtest))
g = tree.export_graphviz(clf
                        ,feature_names=feature_name
                        ,class_names=target_name
                        ,filled=True
                        ,rounded=True)
gra = graphviz.Source(g)
#gra.render(r'C:\Users\ASUS\Desktop\decisontree.gv',view=True)

