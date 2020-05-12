import pandas as pd
import numpy as np

insurance = pd.read_csv('insurance.csv')
insurance['sex'] = insurance.sex.replace({'female':1, 'male':0})
insurance['sex'].value_counts()

from sklearn.model_selection import train_test_split
X = insurance.iloc[:, :3].values
Y = insurance.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr.predict(X_test)

import pickle
pickle.dump(lr, open('insurance.pkl', 'wb'))
model = pickle.load(open('insurance.pkl', 'rb'))
print(model.predict([[30, 1, 4]]))