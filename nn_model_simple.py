import torch
import torch.nn as nn # for neural network model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#matplotlib inline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# create a model class that inherits nn.module
class Model(nn.Module):
    #Input Layer (4 features of the flower )
    #Hidden layer 1 (number of neurons)
    #H2
    #output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() #instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)  # Fully connected layer
        self.fc2 = nn.Linear(h1, h2)           # Fully connected layer
        self.out = nn.Linear(h2, out_features) # Output layer

    # Function that moves everything forward
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Activation function for first layer
        x = F.relu(self.fc2(x)) # Activation function for second layer
        x = self.out(x)         # Output
        return x


# pick a manula seed for randomization
torch.manual_seed(41)
# create an instance of model
model = Model()


# load the data
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
#print(my_df)
#change last columns vareity to numbers because ml works better with numbers
my_df['species'] =my_df['species'].replace('setosa', 0.0)
my_df['species'] =my_df['species'].replace('versicolor', 1.0)
my_df['species'] =my_df['species'].replace('virginica', 2.0)


#train test split setx,y
X =my_df.drop('species', axis = 1 )
y =my_df['species']

# Convert these to numpys arrays
X = X.values
y = y.values


# train the model with iris dataset
#train test split
X_train, X_test, y_train , y_test = train_test_split(X, y , test_size=0.2,  random_state= 41 )
# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# set the criterion of the model to measure the error , how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# CHOOSE Adam Optimizer , learning rate (lr  if error doesnt go down after a bunch of iterations (epochs), lower our lr )
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model.parameters)

# train our model
# epoch? (one rum through all the training data in our network )
epochs = 100
losses = []
for i in range(epochs):
    y_pred = model(X_train)  # get predicted results

    # measure the loss/error , gonna be high at first
    loss = criterion(y_pred, y_train)  # predictedvalues vs the y_train

    # keep the track of our losses
    losses.append(loss.item())  # Correct way to append the loss value

    # print every 10 epoch
    if i % 10 == 0:
        print(f'epoch: {i} and loss: {loss}')

    # DO some back propagation: take the error rate of forward propagation and feed it back
    # through the network to fine tune the weights

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting the loss
plt.plot(range(epochs), losses)
plt.ylabel("Loss/Error")
plt.xlabel("Epoch")
plt.show()


# evaluate model on test data set (validate model on test set)
with torch.no_grad(): #Basically turn off back propagation
  y_eval = model.forward(X_test)# X_test are features from our test , y_eval wil be predictio
  loss = criterion(y_eval, y_test) # find the loss or error

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        # will tel us what type of flower class our network thinks it is
        print(f' {i + 1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        # coreect or not
        if y_val.argmax().item() == y_test[i]:
              correct += 1

print(f'We got {correct} correct')


# predict the labels
label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

correct2 = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # Get the name of the predicted and actual flower
        predicted_label_name = label_map[y_val.argmax().item()]
        actual_label_name = label_map[y_test[i].item()]

        # Print the index, the predicted tensor, and the names of the predicted and actual flowers
        print(f'{i+1}.) {str(y_val)} \t {actual_label_name} \t {predicted_label_name}')

        # Check if the prediction is correct
        if y_val.argmax().item() == y_test[i]:
            correct2 += 1

print(f'We got {correct2} correct')

#find accuracy of the model

# true labels
y_true = [2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 2, 0, 0, 1, 1, 1, 1, 0, 1]

# predicted labels
y_pred = [2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 2, 0, 0, 1, 1, 1, 1, 0, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate a classification report
report = classification_report(y_true, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica'])

print(f'Accuracy: {accuracy}')
print(report)

# fit new data in the dataset
new_iris = torch.tensor([4.7, 3.2 , 1.3 , 0.2])

# Define the class names
label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Predict the class of the new iris sample
new_iris = torch.tensor([4.7, 3.2 , 1.3 , 0.2])
with torch.no_grad():
    predictions = model(new_iris)
    predicted_class_index = predictions.argmax().item()
    predicted_class_name = label_map[predicted_class_index]

print(f"The model predicts that the iris is: {predicted_class_name}")


 