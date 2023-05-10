import numpy as np
from scipy.stats import mode
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
 
#Distancia Euclideana
def eucledianDist(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist
 
#KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def train(self, X, y,*args):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        #Haya las distancias
        distances = [eucledianDist(x, x_train) for x_train in self.X_train]
    
        #K mas cercano
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    

if __name__ == '__main__':
    cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    plt.figure()
    plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
    plt.show()


    clf = KNN(k=5)
    clf.train(X_train, y_train)
    predictions = clf.predict(X_test)

    print(predictions)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)