#SVM problem


#Classifying horses and mules  with height and weight as 
#variables
setwd("C:/Users/thoma/Desktop/Master")


#1.Make sure you have your libraries. The e1071 library has SVM algorithms built in. Create the support vectors using the library. 
install.packages("readr")
 
install.packages("e1071")
library(e1071)
library(readr)
library(kernlab)
#library(e1071)
#2.Import the data set 
# Read the data using read.csv2
animal_data <- read.csv2("horse_muel_classification.csv", stringsAsFactors = FALSE, dec = ",", header = TRUE)

# Display the first few rows
head(animal_data)
str(animal_data)
View(animal_data)

#create an is horse indicator variable 
# that shows which variable is horse and which is muel 
ishorse <-  c(rep(-1,10), rep(+1,10))

#create data frame for performing svm

my.data <-data.frame(Height = animal_data['Height'],
                     Weight = animal_data['Weight'],
                     animal = as.factor(ishorse)) 

#view the created data frame 
my.data 
#library(e1071)
#install.packages("e1071")
#If the data is two-dimensional or three-dimensional, it will be easier to plot. 
#plot the data
plot(my.data[,-3],col=(3)/2, pch=19); abline(h=0,v=0,lty =3)

#perform svm by calling the svm method and passing the parameters
#for library(e1071)

svm.model <- ksvm(animal ~ .,
                  data = my.data,
                  type = 'C-classification',
                  kernel = 'linear',
                  scale = FALSE)


#svm.model <- ksvm(animal ~ .,
   #              data = my.data,
 #                type = 'C-svc',
    #             kernel = 'vanilladot',
    #             scale = FALSE)
#summary(svm.model)
#4.Use the trained model to classify new values. 
#We should have a training set and a test set. 
#Then, ingest the new data. For our example, we’re going to use the whole dataset to train the algorithm and then see how it performs. 

#show the support vectors
points(my.data[svm.model$index, c(1,2)], col="orange", cex =2 )


#get parameters of the hyperplane
w <- t(svm.model$coefs) %*% svm.model$SV
b <- -svm.model$rho

#in this 2d case the hyperplane is the line w[1,1]*x1 + w[1,2]*2+b = 0
abline(a=-b/w[1,2], b=-w[1,1]/w[1,2] , col = "blue", lty = 3 )



#5.Once you see how it performs, the algorithm will decide whether the image is a horse or a mule.


#new data - mule,horse , mule 
observations <-  data.frame(Height=c(67,121,100) , Weight=c(121, 190,100))

#plot the data 

plot(my.data[,-3], col=(ishorse+3)/2 , pch=19, xlim=c(0,250), ylim=c(0,250))
abline(h=0, v=0, lty=3)
points(observations[1,], col="green", pch=19)
points(observations[2,], col="blue", pch=19)
points(observations[3,], col="darkorange", pch=19)
abline(a=-b/w[1,2], b=-w[1,1]/w[1,2], col="blue", lty=3)

#veriy the results 
predict(svm.model,observations)