import numpy as np
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Classification:

    def h(self,x):
        return x @ self.w + self.b
    

    def logistic_predict(self,x):
        return (1 + np.exp(-self.h(x)))**-1
    

    def logistic_loss(self,x,y):

        eps = 0.0000000000001

        class_0_loss = np.dot(1 - y,np.log2(1 - self.logistic_predict(x) + eps))
        class_1_loss = np.dot(y,np.log2(self.logistic_predict(x) + eps))
    
        return (class_0_loss + class_1_loss)/(-len(y))
    
    def logistic_loss_derivatives(self,x,y):
        dw = (1/len(y))*( (y - self.logistic_predict(x)) @ -x )
        db = np.sum(y - self.logistic_predict(x))*(-1/len(y))

        return  dw,db

    def update_parameters(self,derivates):
        dw,db = derivates
        self.w -= self.alpha*dw
        self.b -= self.alpha*db

    def plot_losses(self):
        plt.plot([i for i in range(len(self.loss))],self.loss,label='training')
        plt.plot([i for i in range(len(self.loss_validate))],self.loss_validate,label='validation')
        plt.legend()
        plt.show()

    def train(self,x,y,x_val,y_val):
        np.random.seed(2001)
        self.w = np.array([np.random.rand() for i in range(len(x.T))])

        L = self.Loss(x,y)
        self.loss = [L]
        self.loss_validate = []

        for _ in range(self.epochs):
            self.loss_validate.append(self.Loss(x_val,y_val))
            der = self.Loss_derivate(x,y)
            self.update_parameters(der)
            L = self.Loss(x,y)
            self.loss.append(L)
    
    def plot2d(self,x1,x2,y):
        if len(self.w) == 2:
            x_ = np.linspace(0,1,10)
            y_ = (-self.b - self.w[0]*x_)/(self.w[1])
            plt.plot(x_,y_)
            plt.scatter(x1,x2,c = ["red" if elem else "blue" for elem in y])
            plt.show()
        else:
            print("Error: el numero de dimensiones debe ser 2")
    
    def __init__(self,alpha,epochs,model='logistic'):

        functions_dictionary = {'logistic': (self.logistic_predict,self.logistic_loss,self.logistic_loss_derivatives)}

        self.Predict,self.Loss,self.Loss_derivate = functions_dictionary[model]


        self.model = model
        self.w = []
        self.b = 0
        self.alpha = alpha
        self.epochs = epochs
        self.loss = []
        self.loss_validate = []
    



def normVector(X):
    scaler = MinMaxScaler()
    new_x = np.array(X).reshape(-1, 1)

    return scaler.fit_transform(new_x).reshape(-1)

def splitData(X,Y):

    X_train, X_, Y_train, Y_ = train_test_split(X,Y,test_size = 0.3,random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_,Y_,test_size = 20/30,random_state=42)

    # return X_train,X_,Y_train,Y_
    return X_train,X_val,X_test,Y_train,Y_val,Y_test


data = pd.read_csv('./testing.csv')



X0 = normVector(np.array(data['C1']))
X1 = normVector(np.array(data['C2']))

X = np.transpose(np.array([X0,X1]))
Y = normVector(np.array(data['Clase']))

X_train,X_val,X_testing,Y_train,Y_val,Y_testing = splitData(X,Y)

logistic_regresion = Classification(0.15,10000)

logistic_regresion.train(X_train,Y_train,X_val,Y_val)
logistic_regresion.plot_losses()
# logistic_regresion.plot2d(X.T[0],X.T[1],Y)
# print(Y.shape)

