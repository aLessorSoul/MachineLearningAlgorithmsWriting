import numpy as np
import pandas as pd

df=pd.read_csv(r'data/iris.csv')
df.head()

dic={}
num=1
for i in df.iloc[:,4].unique():
    dic[i]=num
    num+=1


X=df.iloc[:,:3].values
y=df.iloc[:,3].values#=='Iris-setosa'

class LRmodel:
    def __init__(self):
        self.parameters=None
        self.method=None
    
    def fit(self, X, y):
        self.parameters=np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        
    def predict(self, X):
        return np.dot(X, parameters)
lm=LRmodel()
lm.fit(X,y)
lm.predict(X)

lm.parameters