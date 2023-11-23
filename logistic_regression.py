
#1) Design model (input, output size, forward pass)
#2) Construc loss and optimizer
#3) Training loop
# - forward pass: compute prediction and loss
# - backward pass: gradients
# - update weights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# change number of itterations , epochs , lr, optimizer
# to have another accuracy
# 0) prepare the data
bc = datasets.load_breast_cancer()
#iris = datasets.load_iris()
#wn = datasets.load_wine()
X , y = bc.data, bc.target
#X , y = iris.data, iris.target
#X , y = wn.data, wn.target


n_samples, n_features = X.shape
#print(bc)
#print(X , y)
print(' samples: ', n_samples)
print(' features: ', n_features)

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2 , random_state = 1234)

#scale
sc = StandardScaler() # for logistic regression only
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)



# 1) model
#f  = wx +b , sigmoid function
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()  # Instantiate the loss function properly
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients at the start of each epoch

    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # Backward pass
    loss.backward()

    # Updates
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')

# Accuracy of the model
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')