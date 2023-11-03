#import scikit-learn dataset library

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


#Load dataset
cancer = datasets.load_breast_cancer()
print(cancer)

#print the names of the 13 features
print("Features:", cancer.feature_names)
print("-------------------------")
#print the label type of cance('malignant', 'benign')
print("Labels: ", cancer.target_names)

#shape of the data
print(cancer.data.shape)

# top 5 records of data
print(cancer.data[0:5])

print(cancer.target)

# split the data into training set and test set
X_train, X_test, y_train , y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109) # 70% training and 30% test

# Generating the model
#create the svm modifier
clf = svm.SVC(kernel='linear')
clf2 = svm.SVC(kernel='rbf')
clf3= svm.SVC(kernel= 'poly')
clf4= svm.SVC(kernel= 'sigmoid')


#train the model using the training sets
clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)

#predict the response for the test dataset
y_pred = clf.predict(X_test)
y_pred2 = clf.predict(X_test)
y_pred3= clf3.predict(X_test)
y_pred4= clf4.predict(X_test)


# model accuracy how often is the classifier correct ?

print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Accuracy (rbf):", metrics.accuracy_score(y_test,y_pred2))
print("Accuracy (poly) :", metrics.accuracy_score(y_test,y_pred3))
print("Accuracy4 (sigmoid) :", metrics.accuracy_score(y_test,y_pred4))

#Accuracy: 0.9649122807017544
#Accuracy2: 0.9649122807017544
#Accuracy3: 0.9181286549707602
#Accuracy4: 0.39766081871345027

# check for precision and recall
#for linear
print("Precision:" , metrics.precision_score(y_test,y_pred))
print("Recall" , metrics.recall_score(y_test,y_pred))

#for rbf
print("Precision (rbf):" , metrics.precision_score(y_test,y_pred2))
print("Recall (rbf): " , metrics.recall_score(y_test,y_pred2))

#for polynomial
print("Precision (poly) :" , metrics.precision_score(y_test,y_pred3))
print("Recall (poly) :" , metrics.recall_score(y_test,y_pred3))

#for sigmoid
print("Precision (sigmoid) :" , metrics.precision_score(y_test,y_pred4))
print("Recall (sigmoid) : " , metrics.recall_score(y_test,y_pred4))


#Recall 0.9629629629629629
#Precision (rbf): 0.9811320754716981
#Recall (rbf):  0.9629629629629629
##Precision (poly) : 0.8852459016393442
#Recall (poly) : 1.0
#Precision (sigmoid) : 0.5210084033613446
#Recall (sigmoid) :  0.5740740740740741
