import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

my_data=pd.read_csv("cancerdata.csv")
my_data["diagnosis"]=my_data["diagnosis"].map({"M":1,"B":0})
my_data=my_data.drop(['id'],1)


df_majority=my_data[my_data.diagnosis==0]
df_minority=my_data[my_data.diagnosis==1]

df_minority_upsampled=resample(df_minority,replace=True,n_samples=357,random_state=123)
my_data_upsampled=pd.concat([df_majority,df_minority_upsampled])

my_data_upsampled=my_data_upsampled.iloc[:,0:11]

my_data_upsampled=my_data_upsampled.drop(["perimeter_mean"],1)


X=my_data_upsampled.drop(['diagnosis'],1)
y=my_data_upsampled['diagnosis']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state= 123)


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(accuracy_score(y_test,y_pred))


filename = 'breast_cancer_model.pkl'
pickle.dump(knn, open(filename, 'wb'))