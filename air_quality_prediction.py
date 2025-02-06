#description
#the dataset contains information regarding airpollution.Contains details regarding
#temperature,humidity,amount of NO2,CO etc and target variable air quality.The dataset
# #is used for creating models to predict air quality. 
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#loading data
ap=pd.read_csv("E:\data science\dataset\\updated_pollution_dataset.csv")
print(ap.head(1))
#getting info
print(ap.info())
#finding null values and duplicates
print(ap.isna().sum())
print(ap.duplicated().sum())
#changing datatype
ap['Air Quality']=ap['Air Quality'].astype('string')
#plotting outliers
sns.boxplot(data=ap)
plt.xticks(rotation=90)
#assiging values to independant and dependant variables
x=ap.iloc[:,0:9]
y=ap["Air Quality"]
#onehot encoding and label encoding
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
#splitting data to train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y_encoded,test_size=.2,random_state=0)
# perfroming standard scaler
from sklearn.preprocessing import StandardScaler
stx= StandardScaler()
X_train= stx.fit_transform(x_train)
X_test= stx.transform(x_test)
# importing classification model
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"PredictionData":y_pred})
print(df2)
#checking accuracy
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', (accuracy*100))
#boosting model
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model.fit(x_train, y_train)
y_pred= classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', (accuracy*100))