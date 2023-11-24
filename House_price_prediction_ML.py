

#EXPLORE THE DATA
#import libraries
import matplotlib.pyplot  as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

data = pd.read_excel("HousePricePrediction.xlsx")
print(data)
print(data.info())
# Printing first 5 records of the dataset
print(data.head(5))
print("\nSummary Statistics:")
print(data.describe())

# Count of non-null values in each column
#print("\nCount of non-null values in each column:")
#print(data.count())

# Count of unique values in each column
#print("\nCount of unique values in each column:")
#print(data.nunique())

#shape of  the dataset
print(data.shape)

#DATA PREPROCESSING
#Now, we categorize the features depending on their
# datatype (int, float, object)
#and then calculate the number of them.
#Categorical variables
cat_var =  (data.dtypes == 'object')
cat_var_cols = list(cat_var[cat_var].index)
data['Exterior1st'].unique()
data['MSZoning'].unique()
data['LotConfig'].unique()
data['BldgType'].unique()
#classes of the categorical variables
print('Exterior1st:',data['Exterior1st'].unique())
print('MSZoning:',data['MSZoning'].unique())
print('LotConfig:',data['LotConfig'].unique())
print('BldgType:', data['BldgType'].unique())


#print("Which variables are categorical:", cat_var)
#print("Categorical variables:", cat_var_cols)
#print("Number of Categorical variables:", len(cat_var_cols) )


#interger variables
int_var = (data.dtypes == 'int64')
int_var_cols = list(int_var[int_var].index)
#print("Number of integer variables:",  int_var)
#print("Integer variables:", int_var_cols)
#print("Number of integer variables:", len(int_var_cols))

#float variables
float_var = (data.dtypes == 'float')
float_var_cols = list(float_var[float_var].index)
#print("Number of float variables:",  float_var)
#print("float variables:", float_var_cols)
#print("Number of float variables:", len(float_var_cols))

print("float variables:", float_var_cols)
print("Number of float variables:", len(float_var_cols))
print("Integer variables:", int_var_cols)
print("Number of integer variables:", len(int_var_cols))
print("Categorical variables:", cat_var_cols)
print("Number of Categorical variables:", len(cat_var_cols) )

#EDA ( Exploratory Data Analysis)
#USE HEATMAP TO VIEW THE DATA
# Select only numerical columns
numerical_data = data.select_dtypes(include=['float64', 'int64'])
# Calculate correlation
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_matrix,
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.show()

#FOR THE CATEGORICAL VARIABLES
unique_values = []
for col in cat_var_cols:
    unique_values.append(data[col].unique().size)
plt.figure(figsize=(6,6))
plt.title('No. unique values of Categorical Feature')
plt.xticks(rotation = 90)
sns.barplot(x=cat_var_cols,y=unique_values)
plt.show()

#find out the actual count of each category
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in cat_var_cols:
    y = data[col].value_counts()
    plt.subplot(5, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.show()

#DATA CLEANING
#DROP ID
data.drop(['Id'],
          axis=1,
          inplace = True)
print(data)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)
# too many in salesprice

data['SalePrice'] = data['SalePrice'].fillna(
  data['SalePrice'].mean())

missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)

new_data = data.dropna()
print(new_data)

new_data.isnull().sum()
print(new_data.isnull().sum())


#ONEHOTENCODER - FOR LABEL CATEGORICAL FEATURES
# convert categorical to binary
cat_var =  (new_data.dtypes == 'object')
cat_var_cols = list(cat_var[cat_var].index)
print("Categorical variables:")
print(cat_var_cols)
print("Nο. of Categorical variables:", len(cat_var_cols) )


#ONEHOTENCODER
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_data[cat_var_cols]))
OH_cols.index = new_data.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_data.drop(cat_var_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


#SPLITING DATASET INTO TRAINING AND TESTING
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

print(X,Y)
#split the training set into training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y , train_size=0.8, random_state=0)

#MODEL AND ACCURACY
#SVM, RF, LR 3 models
#As loss we use mean absolute percentage error

#SVM
model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print("SVM accuracy:",mean_absolute_percentage_error(Y_valid,Y_pred))


#Random Forest  Regressor
model_RFR = RandomForestRegressor(n_estimators=30)
model_RFR.fit(X_train,Y_train)
Y_pred = model_RFR.predict(X_valid)

print("Random Forest Regressor accuracy:", mean_absolute_percentage_error(Y_valid,Y_pred))

#Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train,Y_train)
Y_pred = model_LR.predict(X_valid)

print("Linear Regression accuracy:", mean_absolute_percentage_error(Y_valid,Y_pred))

#Catboost
cb_model = CatBoostRegressor()
cb_model.fit(X_train,Y_train)
preds = cb_model.predict(X_valid)

cb_r2_score = r2_score(Y_valid,preds)
print(cb_r2_score)