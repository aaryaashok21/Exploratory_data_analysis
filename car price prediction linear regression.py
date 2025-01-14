# description
# This dataset contains car details to model the price of cars with the available independent variables.
# It will be used by the management to understand how exactly the prices vary with the 
# independent variables. They can accordingly manipulate the design of the cars, the business
# strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the 
# pricing dynamics of a new market.The dataset contains different columns such as carname
# engine type,engine size etc.
# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# loading dataset
df=pd.read_csv("E:\data science\dataset\CarPrice_Assignment.csv")
print(df.head(3))
# getting ifo using info()
print(df.info())
# dropping unwanted columns using drop()
df1=df.drop(columns=["car_ID", "symboling",'aspiration','enginelocation'])
print(df1)
# checking null values and duplicates
print(df1.isna().sum())
print(df1.duplicated().sum())
# changing datatype
df1['CarName']=df1['CarName'].astype('string')
df1['fueltype']=df1['fueltype'].astype('string')
df1['doornumber']=df1['doornumber'].astype('string')
df1['carbody']=df1['carbody'].astype('string')
df1['drivewheel']=df1['drivewheel'].astype('string')
df1['enginetype']=df1['enginetype'].astype('string')
df1['fuelsystem']=df1['fuelsystem'].astype('string')
df1['cylindernumber']=df1['cylindernumber'].astype('string')
# finding correlation 
print(df1.corr(numeric_only=True))
#finding outliers 
sns.boxplot(data=df1)
plt.xticks(rotation=90)
# Remove outliers directly in the original DataFrame
df1 = df1[df1['price'] <= 30000]
print(df1)
df1.reset_index(drop=True, inplace=True)
# checking if outliers are removed 
sns.boxplot(data=df1)
plt.xticks(rotation=90)
# splitting independant and dependant variables
x=df1.iloc[:,:-1]
print(pd.DataFrame(x))
y=df1.iloc[:,-1]
print(pd.DataFrame(y))
# finding unique values in  independant variables
print(df1['CarName'].nunique())
print(df1['doornumber'].nunique())
print(df1['enginetype'].nunique())
print(df1['carbody'].nunique())
print(df1['fueltype'].nunique())
print(df1['fuelsystem'].nunique())
print(df1['cylindernumber'].nunique())
print(df1['drivewheel'].nunique())
# model creation
# performing onehot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorical_columns = ['CarName', 'doornumber', 'enginetype', 'carbody', 
'fueltype', 'fuelsystem', 'cylindernumber', 'drivewheel']
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), categorical_columns)], remainder='passthrough')
X_encoded = column_transformer.fit_transform(df1)
# model creation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_encoded,y,test_size=.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
x_pred=regressor.predict(x_train)
y_pred=regressor.predict(x_test)
# tesing accuracy 
from sklearn import metrics
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred))) 
# checking r2score
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')

