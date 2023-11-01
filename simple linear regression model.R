#code in r to learn statistical analysis
#simple linear regression model
# Load necessary libraries
install.packages("ggplot2")
library(ggplot2)


# 1. Load the Dataset
data(mtcars)

# 2. Explore the Data
head(mtcars)
View(mtcars)
str(mtcars)
summary(mtcars)
summary(mtcars$mpg)
summary(mtcars$wt)

# 3. Create a Scatter Plot
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(title = "Scatter Plot of MPG vs Weight",
       x = "Weight",
       y = "Miles per Gallon")

# 4. Fit a Linear Regression Model

model <- lm(mpg ~ wt, data = mtcars)

# 5. Summarize the Model
summary(model)

# 6. Visualize the Regression Line
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Linear Regression of MPG vs Weight",
       x = "Weight",
       y = "Miles per Gallon")

# 7. Make Predictions
new_data <- data.frame(wt = c(3, 4, 5))  # New data for prediction
predictions <- predict(model, newdata = new_data)
print(predictions)
head(predictions)
str(predictions)
summary(predictions)





#gam model the same dataset 
# Load necessary libraries
install.packages("mgcv")
library(mgcv)
install.packages("ggplot2")
library(ggplot2)

# 1. Load the Dataset
data(mtcars)

# 2. Explore the Data
head(mtcars)
summary(mtcars$mpg)
summary(mtcars$wt)
summary(mtcars$cyl)

# 3. Create Scatter Plots
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(title = "Scatter Plot of MPG vs Weight",
       x = "Weight",
       y = "Miles per Gallon")

ggplot(mtcars, aes(x = cyl, y = mpg)) +
  geom_point() +
  labs(title = "Scatter Plot of MPG vs Number of Cylinders",
       x = "Number of Cylinders",
       y = "Miles per Gallon")

# 4. Fit a GAM
gam_model <- gam(mpg ~ s(wt) + s(cyl, k = 3), data = mtcars)

# 5. Summarize the GAM
summary(gam_model)

# 6. Visualize the Smooth Terms
plot(gam_model, pages = 1)

# 7. Make Predictions
new_data <- data.frame(wt = c(3, 4, 5), cyl = c(4, 6, 8))  # New data for prediction
predictions <- predict(gam_model, newdata = new_data)
print(predictions)
