# description
#This project analyzes student data to predict grades using a classification model. The dataset includes student details such as age, sex, high school type, and scholarships.
#The target variable is grade. The project utilizes Python libraries like Pandas, NumPy, Seaborn, Matplotlib, and Scikit-learn for data analysis and model training.
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# loading data
data=pd.read_csv("E:\data science\dataset\student.csv")
print(data.head(1))
# getting info
print(data.info())
# cleaning data
data = data.drop('Unnamed: 0', axis=1)
print(data.columns)
print(data.isna().sum())
print(data.duplicated().sum())
# finding outliers
sns.boxplot(data=data)
plt.xticks(rotation=90)
#no outliers found
# changing datatype
data['Sex']=data['Sex'].astype('string')
data['High_School_Type']=data['High_School_Type'].astype('string')
data['Scholarship']=data['Scholarship'].astype('string')
data['Additional_Work']=data['Additional_Work'].astype('string')
data['Sports_activity']=data['Sports_activity'].astype('string')
data['Transportation']=data['Transportation'].astype('string')
data['Attendance']=data['Attendance'].astype('string')
data['Reading']=data['Reading'].astype('string')
data['Notes']=data['Notes'].astype('string')
data['Listening_in_Class']=data['Listening_in_Class'].astype('string')
data['Project_work']=data['Project_work'].astype('string')
data['Grade']=data['Grade'].astype('string')
# Assigning independant and dependant varibale to x and y 
x=data.iloc[:,:14]
print(x)
y=data['Grade']
print(y)
# encoding categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cols=['Sex','High_School_Type','Scholarship','Additional_Work','Sports_activity',
      'Transportation','Attendance','Reading','Notes','Listening_in_Class','Project_work','Grade']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),cols)], remainder='passthrough')
x_encoded=ct.fit_transform(data)
# splitting data to training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=.2,random_state=0)
# importing standard scaler
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
# model creation
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)
print(y_pred)
print("Prediction comparison")
predict_1=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
print(predict_1.to_string())
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', (accuracy*100))
# boosting model
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model.fit(x_train, y_train)
Y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, Y_pred)
print('Accuracy:', (accuracy*100))
