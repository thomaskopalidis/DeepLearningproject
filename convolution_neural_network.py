# classify imagess
# MNIST DATASET WITH CNN MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

#%matplotlib inline

# Use a directory within your home directory for dataset storage
dataset_root = './home/tkopalid/DAN/cnn_data/MNIST'
# convert the images into 4 d - tensor (mnist image files ) ( # of images, Height , Width , Color)
transform = transforms.ToTensor()
# Train Data
# convert the images into 4 d - tensor (mnist image files ) ( # of images, Height , Width , Color)
transform = transforms.ToTensor()
# Train Data
train_data = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)
#Test data
test_data = datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform)



print(train_data)
print(test_data)



#Create a small batch size for images
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
##############################
# Define Our CNN Model
# Describe convolutional layer and what it is doing (2 convolutional layers)
# This is just an example in the next video we 'll build out the actual model
#conv1 = nn.Conv2d(1, 6, 3, 1)  # 1 -> input images  # 6 -> ouput
#conv2 = nn.Conv2d(6, 16, 3, 1)  # 3 kernel size
#print(conv1)
#print(conv2)

#Grab 1 MNIST record/image:
#for i, (X_Train, y_train ) in enumerate(train_data):
 # break
#print(X_Train)
#x = X_Train.view(1,1,28,28)

# Perform our first convolution
#x =F.relu(conv1(x))# Rectified Linear Unit for our activation function

# 1 single image, 6 is the filters, 26 x26 is the the image
#x.shape
# pass thru the pooling layer
#x = F.max_pool2d(x,2,2) # kernal of 2 and strid of 2
#x.shape # 26/2 =13

# DO the 2nd convolutional layer
#x = F.relu(conv2(x))
#x.shape # Again, we didnt set padding so we lose 2 pixels arount the outside of the image

# pass thru the pooling layer
#x = F.max_pool2d(x,2,2) # kernal of 2 and strid of 2

#print(x.shape )# 11/2 = 5.5 vut we have to round down, because you cant invent data to round up


# model class


##################### create the  model ###################################
# model class
class convolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    #Fully connected layer
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10) # 10 classes of our datasets

  def forward(self,X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    # Second pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    #Re- View to flatten it out
    X = X.view(-1, 16*5*5)  #negative one so that we can vary the batch size

    # Fully connected layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)


#Create an Instance of our model
torch.manual_seed(41)
model = convolutionalNetwork()
print(model)

#Loss function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) # smaller learning rate, longer gonna take to train


start_time = time.time()


#create variables to Track things
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#for loop of epochs
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0



    # train
    for b,(X_train, y_train) in enumerate (train_loader):
      b+=1 # start our batches at 1
      y_pred = model(X_train) # get predicted values from the training set. Not flattened - 2D
      loss = criterion(y_pred, y_train) # how off are we ? Compare the predictions to correct answers in Y_train

      predicted = torch.max(y_pred.data,1)[1] # add the number of correct predictions. Indexed  of the first person
      batch_corr = (predicted == y_train).sum() # how many we got correct from this batch. True = 1 , False = 0 ,sum those up
      trn_corr += batch_corr # keep track as we go along in training

      #update our parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # print out the results
      if b % 600 == 0:
           print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    #test
    with torch.no_grad(): #NO gradient so we dont update our weights and biases with test data
      for b,(X_test, y_test) in enumerate(test_loader):
        y_val = model(X_test)
        predicted = torch.max(y_val.data, 1 )[1] # Adding up correct predictions
        tst_corr += (predicted == y_test).sum() # T=1 F=0 and sum away


    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)





current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total/60} minutes!')