from decision_tree import DT
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


scaler = MinMaxScaler()

data = pd.read_csv('./proyect_dataset/training.csv')
data2 = pd.read_csv('./proyect_dataset/test.csv')
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
data2[data2.columns[:-1]] = scaler.fit_transform(data2[data2.columns[:-1]])


# data[data.columns[-1]] = update_class(data[data.columns[-1]],1,-1)
# data = data.to_numpy()

print(list({elem for elem in data[data.columns[-1]]}))
# print({elem for elem in update_class(data[data.columns[-1]],1,-1)})
# print(type(data['LB']))
# print(len(data['LB']))



X = data.drop(columns='CLASE').to_numpy()
Y = data['CLASE'].to_numpy()

X_test = data2.to_numpy()

clases_list = sorted(list({elem for elem in data[data.columns[-1]]}))
clases_dict = {clases_list[i]: i for i in range(len(clases_list))}


dt = DT(clases_dict)
dt.train(X,Y)

pred = dt.predict(X_test)
for i in range(len(pred)):
    print(i, pred[i])

# print(Counter())



