# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#######
# Load the Iris dataset
######
iris = load_iris()
X = iris.data
y = iris.target
print(X )
print("-----------------")
print(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50 , random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Plot the decision boundary using the first two features
feature1 = 0  # Choose the feature indices you want to plot
feature2 = 1

# Extract the selected features from the dataset
X_subset = X[:, [feature1, feature2]]


# Fit the classifier to the training data
clf.fit(X_train[:, [feature1, feature2]], y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test[:, [feature1, feature2]])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary
x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y, marker='o', s=25)
plt.xlabel(f"Feature {feature1 + 1}")
plt.ylabel(f"Feature {feature2 + 1}")
plt.title("AdaBoost Classifier Decision Boundary")
plt.show()