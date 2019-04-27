#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:41:54 2019

@author: johnpaul
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
'''
check missing value function
'''
def null_table(data):
    print("Training Data Frame")
    print(pd.isnull(data).sum()) 
df=pd.read_csv("/Users/johnpaul/Downloads/heart.csv")
print(null_table(df))
print(df.info())
print(df.head())
'''
Show a number people who have heart disease and don't have.
'''
sns.countplot(x="target", data=df, palette="BuGn_r")
plt.show()
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
'''
Show a number of sex of people in dataset.
'''
sns.countplot(x='sex', data=df, palette="GnBu_d")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()
countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female : {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male : {:.2f}%".format((countMale / (len(df.sex))*100)))
'''
groupby heart disease
'''
print(df.groupby('target').mean())
'''
Show bar graph Heart Disease Frequency for Ages
'''
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6),color=['#F4D03F','#17A589'])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()
'''
Show bar graph Heart Disease Frequency for Sex
'''
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#34495E','#17A589'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
'''
Show scatter plot of Heart Disease occure wit maximun heart rate and age
'''
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="yellow")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()
'''
Show bar graph Heart Disease Frequency for slope of peak
'''
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#B9770E','#5B2C6F' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()
'''
Show bar graph Heart Disease Frequency for Fasting blood sugar
'''
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()
'''
Show bar graph Heart Disease Frequency for type of chest pain
'''
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()
'''
Preprocessing
'''
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
d = pd.get_dummies(df['restecg'], prefix = "restecg")
e = pd.get_dummies(df['ca'],prefix= "ca")
print(df.columns)
frames = [df, a, b, c, d ,e]
df = pd.concat(frames, axis = 1)
print(df.head())
df = df.drop(columns = ['cp', 'thal', 'slope','restecg','ca'])
y = df.target.values
x_data = df.drop(['target'], axis = 1)
'''
Normalize
'''
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
'''
Build Neuro network model and prediction
'''
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=27))
model.add(Dropout(rate=0.5))
model.add(Dense(3,activation='relu'))
model.add(Dense(3,activation='sigmoid'))
model.add(Dense(1,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
sd=model.predict(x_test)
suk=accuracy_score(y_test,sd.round())
print("Test Accuracy of NeuroNet: {:.2f}%".format(suk*100))
'''
Use SVC from sklean and check accuracy score by use testing data
'''
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(x_test,y_test)*100))
'''
Use Random Forest from sklean and check accuracy score by use testing data
'''
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test,y_test)*100))

methods = [ "NeuroNet","SVM", "Random Forest"]
accuracy = [suk*100,svm.score(x_test,y_test)*100,rf.score(x_test,y_test)*100]
colors = ["purple", "green", "red"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Method")
sns.barplot(x=methods, y=accuracy, palette=colors)
plt.show()
