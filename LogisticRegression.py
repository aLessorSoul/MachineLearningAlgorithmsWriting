import numpy as np
import pandas as pd

df=pd.read_csv(r'data/iris.csv')
df.head()

dic={}
num=1
for i in df.iloc[:,4].unique():
    dic[i]=num
    num+=1

X=df.iloc[:,:4].values
y=df.iloc[:,4].values=='Iris-setosa'


parameters = np.random.randn(X.shape[1])
parameters
def predict(X):
    temp = np.exp(np.dot(X, parameters))
    return temp/(1+temp)
p1 = predict(X)
dBeta = np.dot(X.T, y-p1)
diag=np.diagflat(np.multiply(p1, 1-p1))
ddBeta = np.dot(np.dot(X.T, diag), X)
update = np.dot(np.linalg.inv(ddBeta), dBeta)/X.shape[0]
abs(np.mean(update/parameters))

class LogisticRegressionModel:
    def __init__(self):
        self.parameters=None
        self.method=None
    
    def fit(self, X, y, n_iter=10000, method='Newtonian'):
        self.parameters = np.random.randn(X.shape[1])
        old_params = self.parameters
        for n in range(1, n_iter+1):
            p1 = self.predict(X)
            dBeta = np.dot(X.T, y-p1)

            diag=np.diagflat(np.multiply(p1, 1-p1))
            ddBeta = np.dot(np.dot(X.T, diag), X)
            update = np.dot(np.linalg.inv(ddBeta), dBeta)
            if n % 100 == 0:
                if abs(np.mean(self.parameters/old_params-1))<0.00001:
                    print('Converged after {} iterations.'.format(n))
                    return None
            self.parameters -= update
        print('Did not converge, fitting finished after {} iterations.'.format(n))
        
        
    def predict(self, X):
        temp = np.exp(np.dot(X, parameters))
        return temp/(1+temp)
lm = LogisticRegressionModel()
lm.fit(X,y)
np.mean(np.equal(lm.predict(X)>0.5, y))