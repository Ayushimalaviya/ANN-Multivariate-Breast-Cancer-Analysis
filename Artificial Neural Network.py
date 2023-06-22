#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve,auc,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('wisc_bc_ContinuousVar.csv', na_values = '?')
data.shape
data.isnull().sum()
data.head(10)

#Visualization
sns.countplot(x='diagnosis', data=data)#biased classification



X = data.loc[:,data.columns.drop(['diagnosis'])]
y = data['diagnosis']

#to have look if id is important feature or not
rf = RandomForestClassifier() 
#random forest takes care of categorical values, hence y is not labelencoded.
rf.fit(X, y)

rf.feature_importances_

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(40).plot(kind='bar')


#id has very low importance before and after scaling and intuitvely it is not relevant. hence deleting
X = X.loc[:,X.columns.drop(['id'])]

#applied labelencoder to classification column
Encoder = LabelEncoder()
y = Encoder.fit_transform(y) #Malignant = 1 and Benign = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#scaling the continuous and higher values columns into reduced range
sc_feature = StandardScaler()
X_train = sc_feature.fit_transform(X_train.values)
X_test = sc_feature.transform(X_test.values)
print(X_train.shape, X_test.shape)

#after scaling important feature observation
rf1 = RandomForestClassifier() 
rf1.fit(X_train, y_train)

rf1.feature_importances_

feat_importances = pd.Series(rf1.feature_importances_, index=X.columns)
feat_importances.nlargest(40).plot(kind='bar')


# #### Model Development

#Now dense is use to define hidden and output layer separately
ann_model_0 = 'Hidden layer 1 = Relu, Hidden_layer 2 =  Relu, Output_Layer: Sigmoid'
ann0 = keras.models.Sequential()
ann0.add(keras.layers.Dense(5, input_dim=30, activation='relu'))# input_dim inclues column names
ann0.add(keras.layers.Dense(5, activation='relu'))#2nd hidden layer
ann0.add(keras.layers.Dense(1, activation='sigmoid'))#output layer
#optimizer is used for optimal number of weight
#loss refers to loss function (binary_crossentropy is used for binary class)
#metrics accuracy by giving any metrics. I have given auc and accuracy
ann0.compile(optimizer='adam',loss='binary_crossentropy', metrics=[keras.metrics.AUC()])
model0 = ann0.fit(X_train, y_train, epochs=80, batch_size = 10)


ann_model_1 = 'Hidden layer 1 = Relu, Hidden_layer 2 =  Relu, Output_Layer: Relu'
ann1 = keras.models.Sequential()
ann1.add(keras.layers.Dense(5, input_dim=30, activation='relu'))
ann1.add(keras.layers.Dense(5, activation='relu'))
ann1.add(keras.layers.Dense(1, activation='relu'))
ann1.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model1 = ann1.fit(X_train, y_train, epochs=80, batch_size = 10)


ann_model_2 = 'Hidden layer 1 = sigmoid, Hidden_layer 2 =  sigmoid, Output_Layer: Softmax'
ann2 = keras.models.Sequential()
ann2.add(keras.layers.Dense(5, input_dim=30, activation='relu'))
#ann2.add(keras.layers.Dense(5, activation='sigmoid'))
#regularization to reduce the training process.
ann2.add(keras.layers.Dropout(0.2))
ann2.add(keras.layers.Dense(1, activation='softmax'))
ann2.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model2 = ann2.fit(X_train, y_train, epochs=80, batch_size = 10)


ann_model_3 = 'Hidden layer 1 = Sigmoid, Hidden_layer 2 =  Sigmoid, Output_Layer: Sigmoid'
ann3 = keras.models.Sequential()
ann3.add(keras.layers.Dense(5, input_dim=30, activation='sigmoid'))
ann3.add(keras.layers.Dense(5, activation='sigmoid'))
ann3.add(keras.layers.Dense(1, activation='sigmoid'))
ann3.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model3 = ann3.fit(X_train, y_train, epochs=80, batch_size = 10)


# Randomly selected activation function for random epochs which runs model for that many times as  well as batch which divides the data into 10 batches and updates the model 10 times 

# ### Model validation

dic = {ann_model_0:ann0, ann_model_1:ann1, ann_model_2:ann2,ann_model_3:ann3}
for i, j in dic.items():
    print("Activation Function applied on",i,'\n\n')
    y_pred = j.predict(X_test)
    y_pred = (y_pred>0.5)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print("Loss and Accuracy for ANN Model:",j.evaluate(X_test, y_test),'\n\n')
    print('Area Under Curve Score =',roc_auc_score(y_test,y_pred),'\n\n')
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve',linewidth=2)
    plt.ylabel("True Positive Rates")
    plt.xlabel("False Positive Rates")
    plt.title("ROC curve")
    plt.show()

