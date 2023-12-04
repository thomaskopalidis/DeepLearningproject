#IRIS DATASET

#50 SAMPLES , 3 CLASSES ( IRIS SETOSA, IRIS VERSICOLOR, IRISVIRGINICA)
#4 FEATURES -> length , width of sepals and petals

#Column name : Sepal Length , Sepal Width , Petal Length , Petal Width
#Based on the combination of these four features:
#1. I am trying to predict Class of Flower ( Train Model to distinguish the species from each other. )
#1. It is a multivariate data set ie (variation of Iris flowers of three related species ) It can be found on UCI's Machine Learning
#Reprository
#classes
#class 0: Iris setosa (ie , Setosa)
#class 1: Iris virginica( ie, Virginica)
#class 2: Iris versicolor ( ie, Versicolor)


from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import numpy as np
dg = load_digits
dt = load_iris()


print("Type of data: {}".format(type(dt['data'])))
print("Shape of data: {}".format(dt['data'].shape))
print("First five columns: \n {}".format(dt['data'][:5]))
print("Type of target: {}".format(type(dt['target'])))
print("Keys of iris dataset: {}".format(dt.keys()))
print("Target names: {}".format(dt['target_names'])) # y value
print("Feature names: {}".format(dt['feature_names'])) # x value colummn
print("Shape of Target: {}".format(dt['target'].shape))
print("Number of Feature names: {}".format(len(dt['feature_names'])))
X = dt['data']
y = dt['target']
print('Target: \n{}'.format(dt['target']))
print(X.shape)
print(y.shape)

#print(dt)


#TRAIN,SPLIT DATASET 75%, 25% default
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print('X train shape: {}'.format(X_train.shape))
print('X test shape: {}'.format(X_test.shape))
print('y train shape: {}'.format(y_train.shape))
print('y test shape: {}'.format(y_test.shape))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print(knn.fit(X_train,y_train))

y_pred = knn.predict(X_test)
print("Test set prediction: {}".format(y_pred))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print("accuracy", accuracy(y_test, y_pred))
print("Score: {}".format((knn.score(X_test,y_test))))

#TRAIN,TEST SPLIT ---70%, 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 0)
print('X train shape: {}'.format(X_train.shape))
print('X test shape: {}'.format(X_test.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred1 = knn.predict(X_test)
print("Test set prediction: {}".format(y_pred1))
print("accuracy", accuracy(y_test, y_pred1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15,random_state = 0)
print('X train shape: {}'.format(X_train.shape))
print('X test shape: {}'.format(X_test.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred2 = knn.predict(X_test)
print("Test set prediction: {}".format(y_pred2))
print("accuracy", accuracy(y_test, y_pred2))



 