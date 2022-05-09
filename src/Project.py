from calendar import c
from re import A
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
 
#data collection and prpcessing 
#loading the dataset to pandas Dataframe 

loan_dataset= pd.read_csv(r'C:\Users\pc\Documents\Python_101\Clean_DataSet.csv')

#print(loan_dataset.head())

#Print the number of rows and columns 
loan_dataset.shape

#Some statitical measuers
loan_dataset.describe()

#number of missing valuse in each column
loan_dataset.isnull().sum()

#dropping all the missing values
loan_dataset = loan_dataset.dropna()

#--------------------data encoding------------------#
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
loan_dataset.replace({"Gender":{'Male':0,'Female':1}},inplace=True)
loan_dataset.replace({"Married":{'Yes':1,'No':0}},inplace=True)
loan_dataset.replace({"Education":{'Graduate':1,'Not Graduate':0}},inplace=True)
loan_dataset.replace({"Self_Employed":{'Yes':1,'No':0}},inplace=True)
loan_dataset.replace({"Property_Area":{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)

#Replacing the value of +3
loand_dataset= loan_dataset.replace(to_replace='+3',value=4)

#--------------------data Visualization-------------------#

#education and the loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

#matital status & loan status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
#-----------------------------------------------------------#

#seperating the data and lable 
X= loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y= loan_dataset['Loan_Status']

#Train Test Split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

#Training the model:
#Support Vector Machine Model

Classifier= svm.SVC(kernel='linear')

#training the support Vector Machine model
print(Classifier.fit(X_train,Y_train))

#Model Evaluation
#accuracy score on training data
x_train_Prediction=Classifier.predict(X_train)
training_data_accuracy=accuracy_score(x_train_Prediction,Y_train)
print(' Accuracy on training data :',training_data_accuracy)

#accuracy score on test data
x_test_Prediction=Classifier.predict(X_test)
test_data_accuracy=accuracy_score(x_test_Prediction,Y_test)
print(' Accuracy on test  data :',test_data_accuracy)