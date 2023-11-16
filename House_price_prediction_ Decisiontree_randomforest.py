#House Sales in King County, USA
import pandas as pd # data processing
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')
# dataset -> dt
#id -> notation for a house, Date: Date house was sold  Price: Price is prediction target
# Bedrooms: Number of Bedrooms/House,  Bathrooms: Number of bathrooms/House ,  Sqft_Living: square footage of the home
#Sqft_Lot: square footage of the lot,  Floors: Total floors (levels) in house , Waterfront: House which has a view to a waterfront
#View: Has been viewed,  Condition: How good the condition is ( Overall ) ,  Grade: overall grade given to the housing unit, based on King County grading system
#Sqft_Above: square footage of house apart from basement , Sqft_Basement: square footage of the basement, Yr_Built: Built Year
#Yr_Renovated: Year when house was renovated,  Zipcode: Zip, Lat: Latitude coordinate, Long: Longitude coordinate ,S
# qft_Living15: Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
#Sqft_Lot15: lotSize area in 2015(implies-- some renovations)

dt = pd.read_csv("kc_house_data.csv")
print(dt)
print(dt.head())

# see all the columns
print(dt.columns)

print(dt.info())

#finding unique values, how many values each feature has

for value in dt:
    print('For {}, {} unique values present'.format(value,dt[value].nunique()))

#drop the unessecary columns
dt = dt.drop(['id','date'],axis=1)
print(dt.head())
print(dt.info()) # 2 columns less

#DATA VISUALIZATION WITH SEABORN
plt.figure(figsize=(12, 8))
sns.set_context('notebook', font_scale=1.2)
g = sns.pairplot(dt[['sqft_lot', 'sqft_above', 'price', 'sqft_living', 'bedrooms', 'grade', 'yr_built', 'yr_renovated']],
                 hue='bedrooms', height=3)
plt.show()
sns.jointplot(x='sqft_lot', y='price', data=dt, kind='reg', height=3)
sns.jointplot(x='sqft_above', y='price', data=dt, kind='reg', height=3)
sns.jointplot(x='sqft_living', y='price', data=dt, kind='reg', height=3)
sns.jointplot(x='yr_built', y='price', data=dt, kind='reg', height=3)
plt.show()
sns.jointplot(x='bedrooms', y='price', data=dt, kind='reg', height=3)
sns.jointplot(x='yr_built', y='price', data=dt, kind='reg', height=3)
sns.jointplot(x='grade', y='price', data=dt, kind='reg', height=3)
sns.jointplot(x='sqft_lot', y='sqft_above', data=dt, kind='reg', height=3)
plt.show()

#Correlation between variables
plt.figure(figsize=(15,10))
columns =['price','bedrooms','bathrooms','sqft_living','floors','grade','yr_built','condition']
sns.heatmap(dt[columns].corr(),annot=True)
plt.show()

#0.53  bathrooms, 0.31 bedrooms, 0.7 sqft_living, 0.67 year built
#model on the train data
X = dt.iloc[:,1:].values
y = dt.iloc[:,0].values

#Splitting the data into train, test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


################### MULTIPLE LINEAR REGRESSION ##################
mlr = LinearRegression()
mlr.fit(X_train,y_train)
mlr_score = mlr.score(X_test,y_test)
pred_mlr = mlr.predict(X_test)
expl_mlr = explained_variance_score(pred_mlr,y_test)

print(mlr.fit)
print(mlr_score)
print(pred_mlr)
print(expl_mlr)



################### DecisionTreeRegressor ##################

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(X_train,y_train)
tr_regressor.score(X_test,y_test)
pred_tr = tr_regressor.predict(X_test)
decision_score=tr_regressor.score(X_test,y_test)
expl_tr = explained_variance_score(pred_tr,y_test)



################### Random Forest Regression Model ##################

rf_regressor = RandomForestRegressor(n_estimators=28,random_state=0)
rf_regressor.fit(X_train,y_train)
rf_regressor.score(X_test,y_test)
rf_pred =rf_regressor.predict(X_test)
rf_score=rf_regressor.score(X_test,y_test)
expl_rf = explained_variance_score(rf_pred,y_test)

##############  Calculate Model Score ################

print("Multiple Linear Regression Model Score is ",(mlr.score(X_test,y_test)*100))
print("Decision tree  Regression Model Score is ",(tr_regressor.score(X_test,y_test)*100))
print("Random Forest Regression Model Score is ", (rf_regressor.score(X_test,y_test)*100))

#Let's have a tabular pandas data frame, for a clear comparison

models_score =pd.DataFrame({'Model':['Multiple Linear Regression','Decision Tree','Random forest Regression'],
                            'Score':[mlr_score,decision_score,rf_score],
                            'Explained Variance Score':[expl_mlr,expl_tr,expl_rf]
                           })
models_score.sort_values(by='Score',ascending=False)
print(models_score)

