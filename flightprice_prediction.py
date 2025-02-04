# description
# The objective of the study is to analyse the flight booking dataset obtained from “Ease My Trip” website and to conduct various statistical hypothesis tests 
# in order to get meaningful information from it. The 'Linear Regression' statistical algorithm would be used to train the dataset and predict a continuous target variable.
#'Easemytrip' is an internet platform for booking flight tickets, and hence a platform that potential passengers use to buy tickets.
# A thorough study of the data will aid in the discovery of valuable insights that will be of enormous value to passengers.
# The dataset contains various details such as airlines,destination,stops price etc.
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# loading dataset
df=pd.read_csv("E:\data science\dataset\\flight.csv")
print(df.head(5))
# getting info
print(df.info())
# cleaning dataset
# checking null values ,duplicates ,renaming,changing datatype,dropping unwanted columns
print(df.isna().sum())
print(df.duplicated().sum())
df1 = df.drop(columns=['Unnamed: 0'])
# checking outliers
sns.boxplot(data=df1)
plt.xticks(rotation=90)
# removing outliers from orginal dataset
df1 = df1[df1['price'] <= 100000]
print(df1)
#changing datatype
df1['airline']=df1['airline'].astype('string')
df1['source_city']=df1['source_city'].astype('string')
df1['departure_time']=df1['departure_time'].astype('string')
df1['stops']=df1['stops'].astype('string')
df1['arrival_time']=df1['arrival_time'].astype('string')
df1['destination_city']=df1['destination_city'].astype('string')
df1['class']=df1['class'].astype('string')
# statistical analysis using describe()
print(df1.describe())
# finding correlation
print(df1.corr(numeric_only=True))
# plotting heatmap
crmatrix= df1.corr(numeric_only=True)
sns.heatmap(crmatrix,annot=True,cmap="coolwarm",fmt='.2f')
plt.title("correlation heatmap")
plt.show()
# checking uniques values in columns
print(df1['airline'].nunique())
print(df1['source_city'].nunique())
print(df1['arrival_time'].nunique())
print(df1['stops'].nunique())
print(df1['arrival_time'].nunique())
print(df1['destination_city'].nunique())
print(df1['class'].nunique())
# splitting dataset- assigning independant varibales to x and dependant variables to y
x=df1.iloc[:,:10]
print(x)
y=df1['price']
print(y)
# performing one hot encoding 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorical_columns = ['airline','flight','source_city','departure_time','stops','arrival_time','destination_city','class']
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
# checking accuracy
from sklearn import metrics
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# checking r2_score
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')




