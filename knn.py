import numpy as np
from scipy.stats import mode
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KDTree
 
 
#KNN
class KNN:
    def __init__(self,clases, k=3,distance='euclidean'):
        self.k = k
        self.clases = clases
        self.distance = distance


    def train(self, X,Y, *args):
        self.tree = KDTree(X,metric=self.distance)
        self.labels = Y

    def predict(self, X):
        _,indexes = self.tree.query(X,self.k)
        #K mas cercano
        # print(indexes)
        k_nearest_for_every_point = self.labels[indexes] 

        # print(k_nearest)
        # print(X)
        result =   [Counter(k_nearest_point_i).most_common()[0][0] for k_nearest_point_i in k_nearest_for_every_point]

        return result
    
    def probPredict(self,X):
        _,indexes = self.tree.query(X,self.k)
        #K mas cercano
        # print(indexes)
        k_nearest_for_every_point = self.labels[indexes] 
        result = []

        # print(k_nearest)
        # print(X)

        for k_nearest_point_i in k_nearest_for_every_point:
            c = Counter(k_nearest_point_i)
            probs = []
            
            for class_ in self.clases:
                probs.append(c[class_]/c.total())

            result.append(probs)

        return np.array(result)



    

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