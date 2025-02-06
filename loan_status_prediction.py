#Description
#This dataset contains information about individuals applying for loans, including
#personal details age, gender, education, income, employment experience, and
#home ownership, as well as loan-specific features amount, interest rate, intent,
#  and income percentage. It also includes financial attributes like credit history,
#  credit score, and previous defaults. The target variable, loan_status, indicates 
# whether the individual defaulted on the loan or not. This data is typically used 
# for logistic regression to predict the likelihood of loan default based on the 
# provided attributes.
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# loading data
df=pd.read_csv("E:\data science\dataset\loan approval classification.csv")
print(df.head(3))
# cleaning dataset
# finding null values ,duplicates,changing datatypes,renaming column names
print(df.isna().sum())
print(df.duplicated().sum())
# renaming column names
df=df.rename(columns={'cb_person_cred_hist_length':'cred_hist','previous_loan_defaults_on_file':'prev_defaults'})
print(df.info())
# changing datatype
df['person_education']=df['person_education'].astype('string')
df['person_gender']=df['person_gender'].astype('string')
df['person_home_ownership']=df['person_home_ownership'].astype('string')
df['loan_intent']=df['loan_intent'].astype('string')
df['prev_defaults']=df['prev_defaults'].astype('string')
#plotting outliers using boxplot
sns.boxplot(data=df)
plt.xticks(rotation=90)
#as neglible amount of outliers are present data is kept as it is.
#splitting data to independant varibale x and dependant varibale y
x=df.iloc[:,0:13]
print(x)
y=df['loan_status']
print(y)
# encoding categorical values
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
cat_col=['person_gender','person_education','person_home_ownership','loan_intent','prev_defaults']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),cat_col)], remainder='passthrough')
x_encoded=ct.fit_transform(df)
# splitting train test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=.2,random_state=0)
# model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
# fit the model with data
logreg.fit(x_train,y_train)
x_pred=logreg.predict(x_train)
y_pred=logreg.predict(x_test)
# accuracy checking 
from sklearn import metrics
print('mae:',metrics.mean_absolute_error(y_test,y_pred))
print('mse:',metrics.mean_squared_error(y_test,y_pred))
print('rmse:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))