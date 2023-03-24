import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
veri=sns.load_dataset("iris")
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
species=veri.iloc[:,4:5]
species=ohe.fit_transform(species).toarray()
sonuc1=pd.DataFrame(data=species,index=range(150),columns=["setosa","versicolor","virginica"])
sonuc2=veri.iloc[:,0:4]
veri1=pd.concat([sonuc2,sonuc1],axis=1)
from sklearn.model_selection import train_test_split
x=veri1.iloc[:,0:3]
y=veri1.iloc[:,4:7]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_test=sc.fit_transform(y_test)
Y_train=sc.fit_transform(y_train)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
import sklearn.metrics as sm
print(sm.accuracy_score(y_test, y_pred))

